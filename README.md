# Auto Video Montage

Système de montage vidéo automatique piloté par des agents IA. Prend des rushes bruts en entrée et produit un montage exportable, en orchestrant plusieurs agents Claude de façon non-linéaire avec boucle de révision critique.

---

## Architecture

```
┌──────────────────┐
│      START       │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────┐
│          ANALYZER            │
│  ffprobe → scènes            │
│  thumbnails → Claude vision  │
│  sortie : VideoSegment[]     │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐      ┌─────────────────────────┐
│          SCENARIO            │◄─────│        REVISION         │
│  segments + révisions        │      │  CriticFeedback         │
│  → EditPlan (timeline JSON)  │      │  → RevisionInstructions │
└────────┬─────────────────────┘      └────────────▲────────────┘
         │                                         │
         ▼                                         │ score < seuil
┌──────────────────────────────┐                   │ ET iter < max
│           EDITOR             │                   │
│  moviepy exécute EditPlan    │                   │
│  → montage_iter_N.mp4        │                   │
└────────┬─────────────────────┘                   │
         │                                         │
         ▼                                         │
┌──────────────────────────────┐                   │
│           CRITIC             │───────────────────┘
│  évalue EditPlan + keyframes │
│  → score 0.0–1.0 + feedback  │
└────────┬─────────────────────┘
         │ score ≥ seuil OU iter = max
         ▼
┌──────────────────────────────┐
│          QUALITY             │
│  vérifications techniques    │
│  export EDL + FCPXML         │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│          OUTPUT              │
│  montage.mp4                 │
│  sequence.edl                │
│  sequence.fcpxml             │
└──────────────────────────────┘
```

### Agents

| Agent | Modèle | Rôle |
|-------|--------|------|
| **ANALYZER** | claude-haiku-4-5 | Analyse les rushes via ffprobe (métadonnées, détection de scènes) puis vision Claude sur les thumbnails (tags sémantiques, émotion, rôle narratif) |
| **SCENARIO** | claude-sonnet-4-6 | Construit le plan de montage (EditPlan) : ordre des segments, transitions, vitesses, arc narratif |
| **EDITOR** | *(aucun LLM)* | Exécute l'EditPlan via moviepy et produit le fichier mp4 |
| **CRITIC** | claude-sonnet-4-6 | Évalue le plan + 3 keyframes du rendu. Produit un score 0–1 et des notes détaillées |
| **REVISION** | claude-haiku-4-5 | Traduit le feedback du CRITIC en instructions concrètes pour le SCENARIO |
| **QUALITY** | claude-haiku-4-5 | Gate finale : vérifications techniques + export NLE |

### Boucle de révision

```
CRITIC score < seuil  →  REVISION  →  SCENARIO  →  EDITOR  →  CRITIC
                               (max N itérations, configurable)
```

---

## Installation

### Prérequis

- Python 3.11+
- Une clé API Anthropic (console.anthropic.com → API Keys)
- **Pas besoin d'installer ffmpeg** — le binaire est embarqué via `imageio-ffmpeg`

### Setup

```bash
# 1. Cloner / ouvrir le dossier
cd montage-auto

# 2. Créer un environnement virtuel
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer la clé API
cp .env.example .env
# Éditer .env et renseigner ANTHROPIC_API_KEY=sk-ant-...
```

---

## Utilisation

### Interface Streamlit (recommandé)

```bash
streamlit run app.py
```

1. Renseigner la clé Anthropic dans la barre latérale
2. Déposer les fichiers rush (.mp4, .mov, .avi, .mkv)
3. Ajuster le nombre d'itérations max et le seuil de qualité
4. Cliquer **Generate Montage**
5. Télécharger le résultat : mp4, EDL ou FCPXML

### CLI

```bash
# Fichiers individuels
python -m src.main rushes/clip1.mp4 rushes/clip2.mp4

# Tous les fichiers d'un dossier
python -m src.main rushes/ --max-iter 3

# Avec dump de l'état complet en JSON
python -m src.main rushes/ --dump-state
```

---

## Outputs

Tous les fichiers sont générés dans le dossier `output/` :

| Fichier | Description |
|---------|-------------|
| `montage_iter_N.mp4` | Rendu vidéo de chaque itération |
| `<titre>.edl` | EDL CMX 3600 — import DaVinci Resolve |
| `<titre>.fcpxml` | FCPXML 1.10 — import DaVinci Resolve / Final Cut |
| `thumbnails/` | Frames extraites pour l'analyse vision |
| `state_<id>.json` | État complet de l'orchestration (avec `--dump-state`) |

### Importer dans DaVinci Resolve

**EDL** : `Fichier > Importer > Chronologie` → sélectionner le `.edl`  
**FCPXML** : `Fichier > Importer > Chronologie` → sélectionner le `.fcpxml`

> Les deux formats référencent les fichiers sources originaux. DaVinci peut faire un conform à pleine qualité sans repasser par le mp4 rendu.

---

## Structure du projet

```
montage-auto/
├── app.py                    # UI Streamlit
├── requirements.txt
├── .env                      # ANTHROPIC_API_KEY (non versionné)
└── src/
    ├── config.py             # Seuils, modèles, chemins
    ├── models.py             # Contrats Pydantic entre agents
    ├── orchestrator.py       # Machine à états non-linéaire
    ├── export.py             # Générateurs EDL + FCPXML
    ├── main.py               # Entrypoint CLI
    ├── agents/
    │   ├── base_agent.py     # Wrapper Anthropic (structured output)
    │   ├── analyzer.py       # Agent ANALYZER
    │   ├── scenario.py       # Agent SCENARIO
    │   ├── editor.py         # Agent EDITOR (moviepy)
    │   ├── critic.py         # Agent CRITIC
    │   ├── revision.py       # Agent REVISION
    │   └── quality.py        # Agent QUALITY
    └── video/
        ├── probe.py          # ffprobe : métadonnées + détection scènes
        ├── thumbnails.py     # Extraction frames pour vision Claude
        └── editor.py         # Exécuteur moviepy
```

---

## Configuration

Les paramètres principaux sont dans `src/config.py` :

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `QUALITY_THRESHOLD` | `0.70` | Score minimum du CRITIC pour valider le montage |
| `MAX_ITERATIONS` | `3` | Nombre max de cycles CRITIC → REVISION → SCENARIO |
| `TARGET_MONTAGE_DURATION` | `60` | Durée cible du montage en secondes |
| `MAX_SEGMENTS_PER_RUSH` | `10` | Nombre max de scènes extraites par fichier |

---

## Limites (POC)

- Le CRITIC évalue le **plan** et des **keyframes** statiques, pas la vidéo en temps réel
- Les rushes très longs (>10 min) peuvent ralentir l'analyse — ajuster `MAX_SEGMENTS_PER_RUSH`
- Pas de gestion de la musique (le SCENARIO suggère un titre, sans l'intégrer)
- La détection de scènes est basée sur les I-frames ffmpeg, pas sur un modèle de vision dédié
