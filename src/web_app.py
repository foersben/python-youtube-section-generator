import logging
import os
import re
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

# Import refactored modules
from src.core import formatting
from src.core.models.models import Section, SectionGenerationConfig
from src.core.services.section_generation import SectionGenerationService
from src.core.transcript import extract_transcript, extract_video_id

# Set up centralized logging
from src.utils.logging_config import get_logger, setup_logging

setup_logging(
    level="INFO",
    log_file="logs/web_app.log" if not getattr(sys, "frozen", False) else None,
    colored=True,
)

# Suppress noisy Flask/Werkzeug logs in development
logging.getLogger("werkzeug").setLevel(logging.WARNING)

logger = get_logger(__name__)


def _load_dotenv_next_to_executable() -> None:
    """Loads a dotenv file named ".env" located next to the executable if the script is being run as a frozen executable. Otherwise, it will attempt to locate the ".env" file two directories above the current script file. If found, the environment variables defined in the file will be loaded into the system."""

    if getattr(sys, "frozen", False):
        base_dir = Path(sys.executable).parent
    else:
        # Running from source: project root (one level above src)
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
    """Renders the main application interface.

    Returns:
        Flask Response with the rendered index template and cache-control
        headers to force browsers to fetch the latest client-side script.
    """
    from os import getenv

    html = render_template(
        "index.html",
        current_pipeline=getenv("PIPELINE_STRATEGY", "legacy"),
        current_rag=getenv("USE_RAG", "auto"),
    )
    resp = Response(html)
    # Prevent caching so client always receives updated JS after edits
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/generate-sections", methods=["POST"])
def generate_sections() -> Response:
    """Generates YouTube-style section timestamps from a video transcript.

    Processes POST request with video ID and parameters, then:
    1. Extracts the transcript
    2. Generates section timestamps using AI (via SectionGenerationService)
    3. Formats sections for YouTube

    Request Form Parameters:
        video_id: YouTube video ID or URL
        translate_to: (Optional) Language code for translation
        min_sections: Minimum number of sections to generate
        max_sections: Maximum number of sections to generate
        min_title_words: Minimum words in section titles
        max_title_words: Maximum words in section titles

    Returns:
        JSON response with:
        - success: Boolean indicating operation status
        - sections: Formatted sections (if successful)
        - video_id: Processed video ID (if successful)
        - error: Error message (if failed)

    HTTP Status Codes:
        200: Successful operation
        500: Server error during processing
    """

    try:
        # Video ID or URL is optional when providing raw transcript_json
        video_url = request.form.get("video_id", "")
        logger.info(f"Received request for video URL: {video_url}")

        pipeline_strategy = request.form.get("pipeline_strategy")
        if pipeline_strategy:
            os.environ["PIPELINE_STRATEGY"] = pipeline_strategy
            logger.info("Using pipeline strategy: %s", pipeline_strategy)

        rag_mode = request.form.get("rag_mode")
        if rag_mode:
            rag_mode_normalized = rag_mode.lower()
            if rag_mode_normalized in {"always", "auto", "never"}:
                os.environ["USE_RAG"] = rag_mode_normalized
                logger.info("Using RAG mode: %s", rag_mode_normalized)
            else:
                logger.warning("Ignoring invalid RAG mode from request: %s", rag_mode)

        # Parse raw transcript if provided (we'll compute a synthetic video_id)
        raw_transcript_json = request.form.get("transcript_json")
        if raw_transcript_json:
            import hashlib
            import json

            try:
                parsed_temp = json.loads(raw_transcript_json)
            except Exception as e:
                logger.error("Invalid transcript_json provided: %s", e)
                return jsonify({"success": False, "error": "Invalid transcript_json"}), 400

            # Compute a stable id based on the transcript text snippet
            concat = "".join(seg.get("text", "") for seg in parsed_temp[:100])
            video_id = hashlib.md5(concat.encode("utf-8")).hexdigest()[:12]
            logger.info(f"Using synthetic video_id: {video_id} for uploaded transcript")
            transcript_data = parsed_temp
        else:
            if not video_url:
                return (
                    jsonify({"success": False, "error": "Missing video_id or transcript_json"}),
                    400,
                )
            try:
                video_id = extract_video_id(video_url)
                logger.info(f"Extracted Video ID: {video_id}")
            except Exception as e:
                logger.error("Failed to extract video id: %s", e)
                return jsonify({"success": False, "error": "Invalid YouTube URL or ID"}), 400

        translate_to = request.form.get("translate_to", "")
        # Support providing a raw transcript JSON for testing or upload
        raw_transcript_json = request.form.get("transcript_json")

        # Parse section generation parameters
        min_sections = int(request.form.get("min_sections", 10))
        max_sections = int(request.form.get("max_sections", 15))
        min_title_words = int(request.form.get("min_title_words", 3))
        max_title_words = int(request.form.get("max_title_words", 6))

        # Get transcript: allow raw JSON (testing) or fetch from YouTube
        if raw_transcript_json:
            import json

            try:
                transcript_data = json.loads(raw_transcript_json)
            except Exception as e:
                logger.error("Invalid transcript_json provided: %s", e)
                return jsonify({"success": False, "error": "Invalid transcript_json"}), 400
        else:
            transcript_data = extract_transcript(
                video_id=video_id,
                translate_to=translate_to if translate_to else None,
                output_file="./transcript.json",
            )

        # Create configuration for section generation
        generation_config = SectionGenerationConfig(
            min_sections=min_sections,
            max_sections=max_sections,
            min_title_words=min_title_words,
            max_title_words=max_title_words,
            use_hierarchical=True,  # Use hierarchical structure by default
            temperature=0.2,  # Low temperature for consistent output
        )

        # Generate sections using refactored service
        service = SectionGenerationService()
        try:
            sections_list: list[Section] = service.generate_sections(
                transcript=transcript_data,
                video_id=video_id,
                generation_config=generation_config,
            )
        except RuntimeError as e:
            # Likely no LLM provider configured; return helpful error
            logger.exception("LLM generation failed: %s", e)
            return (
                jsonify(
                    {
                        "success": False,
                        "error": str(e),
                        "hint": "Set USE_LOCAL_LLM=true with a local model or configure GOOGLE_API_KEY for Gemini.",
                    }
                ),
                400,
            )

        # Convert Section objects to dict format for formatting
        sections_data = [section.to_dict() for section in sections_list]

        # Post-process titles: replace numeric-only or garbage titles with
        # a cleaned fallback derived from nearby transcript text or a timestamp.
        # Only apply this when titles look truly invalid (not just because they're in English)
        def _has_letter(s: str) -> bool:
            for ch in s:
                if ch.isalpha():
                    return True
            return False

        def _is_valid_title(title: str) -> bool:
            """Check if title looks like a valid section title (not garbage)."""
            # Must have letters
            if not _has_letter(title):
                return False

            # Must be at least 3 characters
            if len(title.strip()) < 3:
                return False

            # Must not be all digits
            if title.isdigit():
                return False

            # Must not contain long digit sequences (like timestamps)
            if re.search(r"\d{3,}", title):
                return False

            # Must not be just punctuation or special chars
            if not re.search(r"[A-Za-z]", title):
                return False

            # Must have at least one space (indicating multiple words) or be a proper noun
            words = title.split()
            if len(words) < 2 and not any(word[0].isupper() for word in words):
                return False

            return True

        # Format for YouTube (text) — robustly handle formatting errors
        try:
            youtube_sections_text = formatting.format_sections_for_youtube(sections_data)
        except Exception:
            logger.exception("Formatting sections failed; falling back to JSON list")
            youtube_sections_text = "\n".join(
                f"{s['start']:.1f}s - {s['title']}" for s in sections_data
            )

        logger.info(f"Successfully generated {len(sections_data)} sections for video {video_id}")

        # Cleanup service resources
        try:
            service.cleanup()
        except Exception:
            logger.exception("Service cleanup failed")

        # Return success
        return (
            jsonify(
                {
                    "success": True,
                    "sections_text": youtube_sections_text,
                    "sections": sections_data,
                    "video_id": video_id,
                    "pipeline_strategy": pipeline_strategy
                    or os.getenv("PIPELINE_STRATEGY", "legacy"),
                    "rag_mode": os.getenv("USE_RAG", "auto"),
                }
            ),
            200,
        )

    except Exception as e:
        logger.exception(f"Error generating sections: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/download-sections", methods=["POST"])
def download_sections() -> send_file:
    """Provides downloadable text file of generated sections.

    Request Form Parameters:
        sections: Formatted section text
        video_id: YouTube video ID

    Returns:
        Text file attachment with section timestamps

    Notes:
        Creates a temporary file that is automatically deleted after send
    """

    sections_text = request.form["sections"]
    video_id = request.form["video_id"]

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as tmp:
        tmp.write(sections_text)
        tmp_path = tmp.name

    logger.info(f"Preparing download for video {video_id}")

    @after_this_request
    def _cleanup(response):
        try:
            import os

            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                logger.debug("Deleted temporary file %s", tmp_path)
        except Exception:
            logger.exception("Failed to delete temporary file %s", tmp_path)
        return response

    return send_file(
        tmp_path,
        as_attachment=True,
        download_name=f"{video_id}_sections.txt",
        mimetype="text/plain",
    )


def format_transcript_for_display(transcript: list[dict[str, Any]]) -> str:
    """Formats transcript data for human-readable display.

    Args:
        transcript: List of transcript segments as dictionaries
          Each dictionary should contain:
          - 'start': Start time in seconds
          - 'text': Transcript text

    Returns:
        Formatted transcript string with timestamps and text
    """

    return "\n".join(f"[{seg['start']:.1f}s] {seg['text']}" for seg in transcript)


if __name__ == "__main__":
    """Main entry point for running the Flask application.

    Starts a development server on port 5000 with debug mode enabled.
    """

    app.run(debug=True, port=5000)
