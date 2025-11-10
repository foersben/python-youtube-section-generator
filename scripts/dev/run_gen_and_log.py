"""Run section generation on ./transcript.json and log detailed output to scripts/gen_run.log

This script is intended as a reproducible verifier: it will
- load transcript.json
- instantiate SectionGenerationService
- call generate_sections()
- call cleanup()
- write structured logs to scripts/gen_run.log
"""
import json
import logging
from pathlib import Path
from datetime import datetime

from src.core.services.section_generation import SectionGenerationService

LOG_PATH = Path(__file__).parent / "gen_run.log"

logger = logging.getLogger("run_gen")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

logger.info("Starting generation run")

transcript_path = Path("./transcript.json")
if not transcript_path.exists():
    logger.error("transcript.json not found at project root")
    raise SystemExit(1)

transcript = json.loads(transcript_path.read_text())
logger.info("Loaded transcript with %d segments", len(transcript))

service = SectionGenerationService()
try:
    start = datetime.utcnow()
    sections = service.generate_sections(transcript=transcript, video_id="cZ9PHPta9v0")
    duration = (datetime.utcnow() - start).total_seconds()
    logger.info("Generated %d sections in %.1f seconds", len(sections), duration)
    for i, s in enumerate(sections[:10]):
        try:
            logger.debug("Section %d: %s", i + 1, s.to_dict())
        except Exception:
            logger.debug("Section %d (raw): %r", i + 1, s)
except Exception as e:
    logger.exception("Generation failed: %s", e)

logger.info("Calling service.cleanup()")
try:
    service.cleanup()
    logger.info("service.cleanup() completed successfully")
except Exception as e:
    logger.exception("service.cleanup() raised: %s", e)

logger.info("Run finished")
print(f"Wrote log to {LOG_PATH}")

