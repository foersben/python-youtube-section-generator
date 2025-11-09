#!/usr/bin/env python3
"""Quick test of improved title generation with translation."""

import os
import sys
import json
from pathlib import Path

# Force translation on
os.environ['USE_TRANSLATION'] = 'true'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.services.section_generation import SectionGenerationService
from src.core.models import SectionGenerationConfig

def main():
    print("=" * 70)
    print("Testing improved title generation with translation pipeline")
    print("=" * 70)
    
    # Load transcript
    transcript_path = Path(__file__).parent.parent / 'transcript.json'
    if not transcript_path.exists():
        print(f"❌ Transcript not found at {transcript_path}")
        return
    
    transcript = json.loads(transcript_path.read_text())
    
    # Use a small subset for faster testing
    transcript_sample = transcript[:300]  # ~5 minutes
    
    print(f"\n✅ Loaded {len(transcript_sample)} transcript segments")
    print(f"Duration: ~{int(transcript_sample[-1]['start']//60)} minutes\n")
    
    # Configure generation
    config = SectionGenerationConfig(
        min_sections=6,
        max_sections=8,
        use_hierarchical=True,
        temperature=0.2
    )
    
    # Generate
    service = SectionGenerationService()
    print("Generating sections (this will take a moment)...\n")
    
    try:
        sections = service.generate_sections(
            transcript=transcript_sample,
            video_id='test_improved',
            generation_config=config
        )
        
        print(f"✅ Generated {len(sections)} sections\n")
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        
        for idx, s in enumerate(sections, 1):
            level_indent = "  " if hasattr(s, 'level') and getattr(s, 'level', 0) == 1 else ""
            mins = int(s.start // 60)
            secs = int(s.start % 60)
            print(f"{idx}. {level_indent}{mins:02d}:{secs:02d} {s.title}")
        
        print("\n" + "=" * 70)
        
        # Cleanup
        service.cleanup()
        print("\n✅ Test completed successfully")
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

