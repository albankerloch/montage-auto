import os
from pathlib import Path

# Thresholds
QUALITY_THRESHOLD = 0.70
MAX_ITERATIONS = 3

# Models
ANALYZER_MODEL = "claude-haiku-4-5-20251001"   # high volume, per-segment vision
SCENARIO_MODEL = "claude-sonnet-4-6"            # needs reasoning
CRITIC_MODEL = "claude-sonnet-4-6"              # needs reasoning
REVISION_MODEL = "claude-haiku-4-5-20251001"    # translation, not creation
QUALITY_MODEL = "claude-haiku-4-5-20251001"     # simple gate

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "output"
THUMBNAILS_DIR = BASE_DIR / "output" / "thumbnails"
RUSHES_DIR = BASE_DIR / "rushes"

# Video settings
TARGET_MONTAGE_DURATION = 60  # seconds, default target
MAX_SEGMENTS_PER_RUSH = 10
THUMBNAIL_TIME_OFFSET = 0.1   # fraction into segment for thumbnail

# Ensure output dirs exist
OUTPUT_DIR.mkdir(exist_ok=True)
THUMBNAILS_DIR.mkdir(exist_ok=True)
RUSHES_DIR.mkdir(exist_ok=True)
