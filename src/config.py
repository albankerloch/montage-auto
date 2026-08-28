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
COMPARATOR_MODEL = "claude-sonnet-4-6"          # comparaison par paires (moteur graphe)

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "output"
THUMBNAILS_DIR = BASE_DIR / "output" / "thumbnails"
RUSHES_DIR = BASE_DIR / "rushes"

# Video settings
TARGET_MONTAGE_DURATION = 60  # seconds, default target
MAX_SEGMENTS_PER_RUSH = 40    # échantillonné uniformément, plus tronqué aux N premières
THUMBNAIL_TIME_OFFSET = 0.3   # fraction into segment for thumbnail

# ── Moteur graphe + solveur + faisceau (src/pipeline.py) ────────────────────
CACHE_DIR = BASE_DIR / "output" / "cache"
ANNOTATE_BATCH_SIZE = 4       # images par appel vision
METRIC_SAMPLES = 3            # points de mesure locale par plan (pleine résolution)
K_PER_PRESET = 2              # solutions distinctes demandées au solveur par intention
SOLVER_TIME_LIMIT_S = 15.0
DEDUPE_THRESHOLD = 0.85       # Jaccard au-delà duquel deux candidats sont redondants
MAX_CANDIDATES = 6            # plafond de candidats soumis au classement LLM
COMPARATOR_MAX_CUTS = 4       # raccords montrés par candidat

# $ / million de tokens (input, output). Indicatif — l'API ne les expose pas,
# à vérifier sur console.anthropic.com/settings/billing si la grille change.
# Un modèle absent de cette table est compté à $0 (coût non estimable, jamais
# deviné) dans le rapport de `src/usage.py`.
MODEL_PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
}

# Ensure output dirs exist
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
THUMBNAILS_DIR.mkdir(exist_ok=True)
RUSHES_DIR.mkdir(exist_ok=True)
