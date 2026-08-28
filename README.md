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

### Créer la timeline directement dans Resolve (API)

Sans passer par un import de fichier : le plan est construit dans le projet
Resolve courant via l'API de scripting (import des médias dans le Media Pool,
`CreateEmptyTimeline` + `AppendToTimeline`, in/out au fps réel de chaque clip,
bornes re-clampées aux limites du média — garde anti-hallucination du plan LLM).

Prérequis : **Resolve Studio ouvert**, projet actif, et
`Préférences > Système > Général > External scripting using` = **Local**.

```bash
# en une passe, à la fin du pipeline
python -m src.main rushes/ --resolve

# ou après coup, depuis le plan JSON exporté par export_all()
python -m src.export_resolve output/<titre>_plan.json
```

Hors console interne de Resolve, exposer le module de scripting :

```bash
# macOS
export RESOLVE_SCRIPT_API="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
# Windows (PowerShell)
$env:RESOLVE_SCRIPT_API="$env:PROGRAMDATA\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting"
```

Limites de l'API (signalées en console, jamais silencieuses) : les fondus et
le retime (`speed_factor ≠ 1`) ne peuvent pas être posés par
`AppendToTimeline` — ils restent à appliquer dans Resolve (ils sont par
ailleurs présents dans l'EDL/FCPXML exportés).

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
    ├── export_resolve.py     # Timeline directe dans Resolve (scripting API)
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

## Deux moteurs

Le dépôt embarque deux implémentations du même problème. La première est
l'originale ; la seconde est ce vers quoi elle a convergé une fois admis que
l'essentiel de ce qu'on demandait au LLM était un problème d'optimisation sous
contraintes, et que la boucle de révision était une recherche dégradée
(faisceau de 1, récompense scalaire bruitée, aucune garantie de monotonie).

| | `--engine loop` (historique) | `--engine graph` (défaut) |
|---|---|---|
| Ordonnancement | machine à états écrite à la main | graphe de dépendances, remonté à la demande |
| Assemblage | SCENARIO (LLM) écrit la timeline | CP-SAT sous contraintes déclarées |
| Exploration | boucle CRITIC → REVISION, séquentielle | faisceau de N intentions, indépendantes |
| Évaluation | score absolu 0–1 contre un seuil de 0.70 | comparaison par paires, sans calibration |
| Sortie | un montage | un classement : livré + alternates |
| Reprise | aucune | propriété du cache |
| Rejouabilité | non | oui (solveur déterministe) |

```bash
python -m src.main rushes/ --engine graph
python -m src.main rushes/ --engine graph --presets punchy,emotional_arc --duration 90
python -m src.main rushes/ --engine graph --explain   # que recalculerait-on ?
python -m src.main rushes/ --engine loop --max-iter 3 # ancien comportement
```

### Le graphe (`src/graph.py`, `src/nodes.py`)

On ne décrit aucun ordre d'exécution, seulement qui dépend de quoi. Chaque nœud
porte une clé `sha256(nom + version + params + clés des dépendances)` ; demander
un artefact matérialise ce qui manque.

```
rush ─┬─ probe ──────────────┐
      ├─ scenes ─┬─ thumbs ──┼─ annot ─┐
      └──────────┘           │         │
                             └─────────┴─ segments ─ candidates ─ ranked ─┬─ render
                                                                          └─ exports
```

Les nœuds par rush sont indépendants entre rushes. Conséquences vérifiées par
`tests/test_pipeline_e2e.py` :

- ajouter un rush ne réanalyse que ce rush ;
- passer la durée cible de 60 à 90 s n'invalide que `candidates` et l'aval —
  jamais les annotations vision, qui sont le poste de coût ;
- un plantage en aval ne perd pas le travail amont ;
- `--explain` affiche le graphe et ce qui serait recalculé.

Le corps des fonctions n'est **pas** haché : c'est `V[...]` dans `src/nodes.py`
qui déclare qu'un calcul a changé de sémantique, comme une migration. Et le
cache impose le déterminisme : un nœud non déterministe rend la valeur relue
différente d'un recalcul, ce qui est la raison pour laquelle le solveur tourne
sur un worker avec une limite en temps déterministe.

### Le solveur (`src/assemble.py`)

Le LLM annote (`quality_score`, rôle, émotion). Le solveur sélectionne et
ordonne : `x[i][p] = 1` si le segment `i` occupe la position `p`.

Contraintes dures : durée dans la bande de tolérance, chaque segment au plus une
fois, positions contiguës, pas deux plans adjacents du même rush, plan
d'ouverture en tête, résolution en queue, ordre chronologique optionnel.
Objectif : qualité, adéquation rôle/position, écart à une courbe d'énergie,
écart au rythme cible, moins un péage par plan (sans lui, une somme de scores
positifs veut toujours plus de plans et sature le plafond de durée).

Chaque contrainte est testable et se discute avec un monteur, contrairement à un
prompt de 400 mots. `Candidate.gap` expose l'écart à la borne supérieure : on
sait ce qu'on ne sait pas.

### Le faisceau (`src/beam.py`)

Cinq intentions — `chronological`, `emotional_arc`, `punchy`, `contemplative`,
`best_of` — sont des `Preset`, pas des prompts : générer le faisceau coûte zéro
token et les solveurs sont indépendants. Le solveur énumère K solutions
distinctes par intention (no-good de diversité), on déduplique au Jaccard, puis
on classe par comparaison par paires.

Le comparateur reçoit les **frames de raccord** (`src/video/cuts.py`) : la
dernière image avant chaque coupe et la première après. L'ancien CRITIC recevait
trois keyframes à 10/50/90 % du rendu — il ne voyait donc jamais une seule
coupe, et jugeait le contenu en croyant juger le montage.

## Tests

```bash
pip install pytest && python -m pytest tests -q
```

31 tests, sans clé API. Les tests de bout en bout utilisent des doubles
déterministes pour les deux agents LLM et des fixtures vidéo générées par
ffmpeg ; ils se skippent si `tests/fixtures/rushes/` est vide.

---

## Limites (POC)

- Le comparateur juge des images fixes de raccord, pas le mouvement : un faux
  raccord sur un travelling lui échappe encore
- Aucune notion de multicam (synchro, choix d'angle sur action simultanée)
- Le coût et la latence ne sont pas instrumentés : `response.usage` n'est lu
  nulle part, donc aucun chiffre de bout en bout n'est disponible
- Les rushes très longs (>10 min) peuvent ralentir l'analyse — ajuster `MAX_SEGMENTS_PER_RUSH`
- **Pas d'audio du tout** : `has_audio` est relevé et jamais lu. C'est le manque
  le plus visible pour un monteur, le rythme d'un montage étant piloté par la
  parole et la musique. La suite naturelle est d'en faire des contraintes du
  solveur (« chaque coupe à ±80 ms d'un onset »), ce qu'un prompt ne sait pas
  garantir et qu'un modèle CP-SAT garantit par construction
- La détection de scènes est basée sur les I-frames ffmpeg, pas sur un modèle de vision dédié
