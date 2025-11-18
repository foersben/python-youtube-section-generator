"""CLI entry point for YouTube transcript section generator.

Uses refactored architecture with SectionGenerationService.
"""

import argparse
import os

from dotenv import load_dotenv

from src.core import formatting
from src.core.models.models import Section, SectionGenerationConfig
from src.core.services.section_generation import SectionGenerationService
from src.core.transcript import extract_transcript
from src.utils.logging_config import get_logger, setup_logging

# Load environment variables once at startup
load_dotenv()

# Set up centralized logging
setup_logging()

logger = get_logger(__name__)


def main() -> None:
    """Main CLI workflow for generating YouTube sections."""
    parser = argparse.ArgumentParser(
        description="Generate YouTube section timestamps from video transcripts"
    )
    parser.add_argument(
        "video_id",
        nargs="?",
        default=os.getenv("DEFAULT_VIDEO_ID", "kXhCEyix180"),
        help="YouTube video ID or URL (default: from DEFAULT_VIDEO_ID env var)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "."),
        help="Output directory for generated files (default: current directory)",
    )
    parser.add_argument(
        "--translate-to",
        default=os.getenv("TRANSLATE_TO", "en"),
        help="Language to translate to for generation (default: en, use 'none' to disable)",
    )
    parser.add_argument(
        "--min-sections",
        type=int,
        default=int(os.getenv("MIN_SECTIONS", "10")),
        help="Minimum number of sections to generate",
    )
    parser.add_argument(
        "--max-sections",
        type=int,
        default=int(os.getenv("MAX_SECTIONS", "15")),
        help="Maximum number of sections to generate",
    )
    parser.add_argument(
        "--no-hierarchical", action="store_true", help="Disable hierarchical section generation"
    )
    parser.add_argument(
        "--pipeline-strategy",
        default=os.getenv("PIPELINE_STRATEGY", "legacy"),
        choices=["legacy", "split"],
        help="Processing pipeline strategy (legacy | split)",
    )

    args = parser.parse_args()

    # Configuration
    VIDEO_ID = args.video_id
    OUTPUT_DIR = args.output_dir
    TRANSLATE_TO = None if args.translate_to.lower() == "none" else args.translate_to

    TRANSCRIPT_FILE = os.path.join(OUTPUT_DIR, "transcript.json")
    SECTIONS_FILE = os.path.join(OUTPUT_DIR, "sections.json")
    YOUTUBE_SECTIONS_FILE = os.path.join(OUTPUT_DIR, "youtube_sections.txt")

    try:
        print("=" * 70)
        print("YouTube Transcript Section Generator")
        print("=" * 70)
        print(f"Video ID: {VIDEO_ID}")
        print(f"Output Directory: {OUTPUT_DIR}")
        print(f"Translation: {TRANSLATE_TO or 'disabled'}")
        print()

        # Step 1: Fetch transcript
        print(f"Fetching transcript for video: {VIDEO_ID}")
        transcript_data = extract_transcript(
            video_id=VIDEO_ID,
            output_file=TRANSCRIPT_FILE,
            translate_to=TRANSLATE_TO,
        )
        print(f"✅ Transcript fetched ({len(transcript_data)} segments)")
        print()

        # Step 2: Configure section generation
        print("Configuring section generation...")
        generation_config = SectionGenerationConfig(
            min_sections=args.min_sections,
            max_sections=args.max_sections,
            min_title_words=3,
            max_title_words=7,
            use_hierarchical=not args.no_hierarchical,
            temperature=0.2,  # Low temperature for consistent output
        )
        os.environ["PIPELINE_STRATEGY"] = args.pipeline_strategy
        print(f"  Sections: {generation_config.min_sections}-{generation_config.max_sections}")
        print("  Title words: 3-7")
        print(f"  Hierarchical: {generation_config.use_hierarchical}")
        print()

        # Step 3: Generate sections using service
        print("Generating section timestamps...")
        service = SectionGenerationService()
        sections_list: list[Section] = service.generate_sections(
            transcript=transcript_data,
            video_id=VIDEO_ID,
            generation_config=generation_config,
        )
        print(f"✅ Generated {len(sections_list)} sections")
        print()

        # Step 4: Convert to dict format for saving
        sections_data = [section.to_dict() for section in sections_list]

        # Save sections to JSON
        import json

        with open(SECTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(sections_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Sections saved to {SECTIONS_FILE}")
        print()

        # Step 5: Format for YouTube
        print("Formatting for YouTube...")
        youtube_format = formatting.format_sections_for_youtube(sections_data)

        # Display sections
        print()
        print("=" * 70)
        print("YouTube-ready Section Timestamps:")
        print("=" * 70)
        print()
        print(youtube_format)
        print()

        # Save formatted sections
        with open(YOUTUBE_SECTIONS_FILE, "w", encoding="utf-8") as f:
            f.write(youtube_format)
        print(f"✅ YouTube-formatted sections saved to {YOUTUBE_SECTIONS_FILE}")
        print()

        # Display hierarchical structure if applicable
        print("=" * 70)
        print("Section Structure:")
        print("=" * 70)
        print(youtube_format)  # Use the same formatted output
        print()
        print("=" * 70)
        print("✅ Processing Complete!")
        print("=" * 70)

        # Cleanup
        service.cleanup()

    except Exception as e:
        print()
        print("=" * 70)
        print("❌ Processing Failed")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
