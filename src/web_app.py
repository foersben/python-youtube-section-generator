import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    after_this_request,
    jsonify,
    render_template,
    request,
    send_file,
)

from src.core.config import config as app_config
from src.core import formatting
from src.core.models.models import Section, SectionGenerationConfig
from src.core.services.section_generation import SectionGenerationService
from src.core.transcript import extract_transcript, extract_video_id
from src.utils.logging_config import get_logger, setup_logging

setup_logging(
    level="INFO",
    log_file="logs/web_app.log" if not getattr(sys, "frozen", False) else None,
    colored=True,
)

logger = get_logger(__name__)

def _load_dotenv_next_to_executable() -> None:
    if getattr(sys, "frozen", False):
        base_dir = Path(sys.executable).parent
    else:
        base_dir = Path(__file__).resolve().parent.parent
    env_path = base_dir / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)

_load_dotenv_next_to_executable()

_frozen = getattr(sys, "frozen", False)
if _frozen:
    _meipass = getattr(sys, "_MEIPASS", None)
    if _meipass:
        base = Path(_meipass)
        tmpl_folder = base / "templates"
        static_folder = base / "static"
    else:
        base = Path(__file__).resolve().parent.parent
        tmpl_folder = base / "src" / "templates"
        static_folder = base / "static"
else:
    base = Path(__file__).resolve().parent.parent
    tmpl_folder = base / "src" / "templates"
    static_folder = base / "static"

app = Flask(
    __name__,
    template_folder=str(tmpl_folder),
    static_folder=str(static_folder),
)

@app.route("/")
def index() -> Response:
    app_config.reload()
    html = render_template(
        "index.html",
        current_pipeline=app_config.pipeline_strategy,
        current_rag=app_config.use_rag,
    )
    resp = Response(html)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp

@app.route("/generate-sections", methods=["POST"])
def generate_sections() -> tuple[Response, int]:
    try:
        video_url = request.form.get("video_id", "")
        logger.info(f"Received request for video URL: {video_url}")

        # Config Updates
        pipeline_strategy = request.form.get("pipeline_strategy")
        if pipeline_strategy:
            os.environ["PIPELINE_STRATEGY"] = pipeline_strategy

        rag_mode = request.form.get("rag_mode")
        if rag_mode:
            rag_mode_normalized = rag_mode.lower()
            if rag_mode_normalized in {"always", "auto", "never"}:
                os.environ["USE_RAG"] = rag_mode_normalized

        app_config.reload()
        logger.info("Active Config -> Pipeline: %s, RAG: %s", app_config.pipeline_strategy, app_config.use_rag)

        raw_transcript_json = request.form.get("transcript_json")
        if raw_transcript_json:
            import hashlib
            import json
            try:
                parsed_temp = json.loads(raw_transcript_json)
            except Exception as e:
                return jsonify({"success": False, "error": "Invalid transcript_json"}), 400
            concat = "".join(seg.get("text", "") for seg in parsed_temp[:100])
            video_id = hashlib.md5(concat.encode("utf-8")).hexdigest()[:12]
            transcript_data = parsed_temp
        else:
            if not video_url:
                return jsonify({"success": False, "error": "Missing video_id"}), 400
            try:
                video_id = extract_video_id(video_url)
            except Exception:
                return jsonify({"success": False, "error": "Invalid YouTube URL"}), 400

        translate_to = request.form.get("translate_to", "")

        min_sections = int(request.form.get("min_sections", 10))
        max_sections = int(request.form.get("max_sections", 15))
        min_title_words = int(request.form.get("min_title_words", 3))
        max_title_words = int(request.form.get("max_title_words", 6))

        if not raw_transcript_json:
            transcript_data = extract_transcript(
                video_id=video_id,
                translate_to=translate_to if translate_to else None,
                output_file="./transcript.json",
            )

        # Metadata Injection for Translation
        if transcript_data:
            # If user supplied a target language, set it explicitly
            if translate_to:
                transcript_data[0]["target_language"] = translate_to
            # Ensure original language is preserved if not already set
            if "original_language_code" not in transcript_data[0]:
                # Fallback if extractor didn't set it (e.g. raw json upload)
                transcript_data[0]["original_language_code"] = "en"

        generation_config = SectionGenerationConfig(
            min_sections=min_sections,
            max_sections=max_sections,
            min_title_words=min_title_words,
            max_title_words=max_title_words,
            use_hierarchical=True,
            temperature=0.2,
        )

        service = SectionGenerationService()
        try:
            sections_list: list[Section] = service.generate_sections(
                transcript=transcript_data,
                video_id=video_id,
                generation_config=generation_config,
            )
        except RuntimeError as e:
            logger.exception("LLM generation failed")
            return jsonify({"success": False, "error": str(e)}), 400

        sections_data = [section.to_dict() for section in sections_list]

        try:
            youtube_sections_text = formatting.format_sections_for_youtube(sections_data)
        except Exception:
            youtube_sections_text = "\n".join(f"{s['start']:.1f}s - {s['title']}" for s in sections_data)

        try:
            service.cleanup()
        except Exception:
            pass

        return (
            jsonify(
                {
                    "success": True,
                    "sections_text": youtube_sections_text,
                    "sections": sections_data,
                    "video_id": video_id,
                    "pipeline_strategy": app_config.pipeline_strategy,
                    "rag_mode": app_config.use_rag,
                }
            ),
            200,
        )

    except Exception as e:
        logger.exception(f"Error generating sections: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/download-sections", methods=["POST"])
def download_sections() -> Response:
    sections_text = request.form["sections"]
    video_id = request.form["video_id"]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as tmp:
        tmp.write(sections_text)
        tmp_path = tmp.name
    @after_this_request
    def _cleanup(response):
        try:
            if os.path.exists(tmp_path): os.remove(tmp_path)
        except Exception: pass
        return response
    return send_file(tmp_path, as_attachment=True, download_name=f"{video_id}_sections.txt", mimetype="text/plain")

def format_transcript_for_display(transcript: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{seg['start']:.1f}s] {seg['text']}" for seg in transcript)

if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") not in {"0", "false", "False"}
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not debug:
        logging.getLogger("werkzeug").setLevel(logging.INFO)
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            resolved_ip = s.getsockname()[0]
            s.close()
        except Exception: resolved_ip = host
        logger.info("Starting Flask app on %s:%d (pid=%s)", host, port, os.getpid())
    app.run(debug=debug, host=host, port=port)
