from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
from core import transcript, sections, formatting
import tempfile
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)


@app.route("/")
def index() -> str:
    """ Renders the main application interface.

    Returns:
        Rendered HTML template for the main page
    """
    return render_template("index.html")


@app.route("/generate-sections", methods=["POST"])
def generate_sections() -> jsonify:
    """Generates YouTube-style section timestamps from a video transcript.

    Processes POST request with video ID and parameters, then:
    1. Extracts the transcript
    2. Generates section timestamps using AI
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
        video_id = transcript.extract_video_id(request.form["video_id"])
        translate_to = request.form.get("translate_to", "")
        section_count_range = (
            int(request.form.get("min_sections", 10)),
            int(request.form.get("max_sections", 15)),
        )
        title_length_range = (
            int(request.form.get("min_title_words", 3)),
            int(request.form.get("max_title_words", 6)),
        )
        enable_speakers = request.form.get("enable_speakers") == "on"

        # Parse known speakers if provided
        known_speakers = None
        if enable_speakers and request.form.get("known_speakers"):
            try:
                known_speakers = json.loads(request.form.get("known_speakers"))
            except json.JSONDecodeError:
                logger.warning("Invalid JSON format for known speakers")

        # Get transcript with speaker recognition
        transcript_data = transcript.extract_transcript(
            video_id=video_id,
            translate_to=translate_to if translate_to else None,
            enhance_with_speakers=enable_speakers,
            known_speakers=known_speakers,
        )

        # Get transcript
        # transcript_data = transcript.extract_transcript(
        #     video_id=video_id, translate_to=translate_to if translate_to else None
        # )

        # Generate sections
        sections_data = sections.create_section_timestamps(
            transcript=transcript_data,
            section_count_range=section_count_range,
            title_length_range=title_length_range,
        )

        # Format for YouTube
        youtube_sections = formatting.format_sections_for_youtube(sections_data)

        return jsonify(
            {"success": True, "sections": youtube_sections, "video_id": video_id}
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/download-sections", methods=["POST"])
def download_sections() -> send_file:
    """ Provides downloadable text file of generated sections.

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

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as tmp:
        tmp.write(sections_text)
        tmp_path = tmp.name

    return send_file(
        tmp_path,
        as_attachment=True,
        download_name=f"{video_id}_sections.txt",
        mimetype="text/plain",
    )


@app.route("/download-transcript", methods=["POST"])
def download_transcript() -> send_file:
    """ Facilitates the downloading of a transcript file in plain text format for a given video.

    This route handles POST requests and reads the transcript text and video ID from
    the request form. It creates a temporary file with the given transcript content
    and provides it for download as a text file.

    Args:
        transcript_text (str): The text of the transcript to be downloaded.
        video_id (str): The identifier of the video associated with the transcript to include in the file name.

    Returns:
        A response object that initiates the download of the temporary transcript text file as an attachment.
    """

    transcript_text = request.form["transcript"]
    video_id = request.form["video_id"]

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as tmp:
        tmp.write(transcript_text)
        tmp_path = tmp.name

    return send_file(
        tmp_path,
        as_attachment=True,
        download_name=f"{video_id}_transcript.txt",
        mimetype="text/plain",
    )


# def format_transcript_for_display(transcript: list[dict[str, Any]]) -> str:
#     """Formats transcript data for human-readable display.
#
#     Args:
#         transcript: List of transcript segments as dictionaries
#           Each dictionary should contain:
#           - 'start': Start time in seconds
#           - 'text': Transcript text
#
#     Returns:
#         Formatted transcript string with timestamps and text
#     """
#
#     return "\n".join(f"[{seg['start']:.1f}s] {seg['text']}" for seg in transcript)


if __name__ == "__main__":
    """Main entry point for running the Flask application.

    Starts a development server on port 5000 with debug mode enabled.
    """

    app.run(debug=True, port=5000)
