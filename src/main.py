from dotenv import load_dotenv
from core import transcript, sections, formatting


# Load environment variables once at startup
load_dotenv()


if __name__ == "__main__":
    # Configuration
    VIDEO_ID = "kXhCEyix180"  # "G63yfp-owY8"  # Replace with your video ID
    TRANSCRIPT_FILE = "transcript.json"
    SECTIONS_FILE = "sections.json"
    YOUTUBE_SECTIONS_FILE = "youtube_sections.txt"

    try:
        print("Fetching transcript...")
        transcript_data = transcript.extract_transcript(
            video_id=VIDEO_ID,
            output_file=TRANSCRIPT_FILE,
            translate_to="en",  # Remove for original language
        )

        print("\nGenerating section timestamps...")
        sections_data = sections.create_section_timestamps(
            transcript=transcript_data, output_file=SECTIONS_FILE
        )

        print("\nYouTube-ready Section Timestamps:")
        youtube_format = formatting.format_sections_for_youtube(sections_data)
        print("\n" + youtube_format)

        with open(YOUTUBE_SECTIONS_FILE, "w") as f:
            f.write(youtube_format)
        print(f"\nYouTube-formatted sections saved to {YOUTUBE_SECTIONS_FILE}")

    except Exception as e:
        print(f"Processing failed: {e}")
