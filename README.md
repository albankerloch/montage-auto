# Auto Video Montage

Montage vidéo assisté. Prend des rushes bruts, en extrait les plans, les mesure
et les annote, puis assemble une timeline exportable vers un NLE.

Le système ne rend pas un montage, il rend un **classement** : le premier
candidat va au monteur, les suivants sont les alternates. Le monteur veto,
choisit, et ce qu'il livre revient dans le graphe.

Le dépôt contient deux moteurs. `--engine graph` est le défaut et ce que décrit
ce README. `--engine loop` est l'implémentation d'origine, conservée pour
comparaison et documentée en fin de document.

---

## Installation

### Prérequis

- Python 3.11+
- Une clé API Anthropic (console.anthropic.com → API Keys)
- **Pas besoin d'installer ffmpeg** — le binaire est embarqué via `imageio-ffmpeg`

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

pip install -r requirements.txt

cp .env.example .env
# Éditer .env et renseigner ANTHROPIC_API_KEY=sk-ant-...
```

---

## Utilisation

```bash
# montage complet, faisceau par défaut
python -m src.main rushes/

# choisir les intentions et la durée cible
python -m src.main rushes/ --presets punchy,emotional_arc --duration 90

# qu'est-ce qui serait recalculé ? (n'exécute rien)
python -m src.main rushes/ --explain

# arbitrage humain : sortir les K candidats sans qu'aucun modèle ne tranche
python -m src.main rushes/ --rank manual

# choisir un candidat, retirer des plans
python -m src.main rushes/ --pick 2 --ban rush_0@12.250,rush_1@3.000
python -m src.main rushes/ --ban-file bans.json

# construire la timeline directement dans Resolve
python -m src.main rushes/ --resolve
```

### Interface Streamlit

```bash
streamlit run app.py
```

> `app.py` pilote encore le **moteur historique** (`--engine loop`) : elle expose
> le seuil de qualité et le nombre d'itérations, qui n'existent plus dans le
> moteur par défaut. Le moteur graphe s'utilise en ligne de commande.

---

## Architecture

On ne décrit aucun ordre d'exécution : on déclare qui dépend de quoi. Chaque
nœud porte une clé `sha256(nom + version + params + clés des dépendances)`, et
demander un artefact matérialise ce qui manque.

Vue d'ensemble, sept étapes :

```
①scenes → ②thumbs → ③{ metrics ∥ annot } → ④segments → ⑤candidates (CP-SAT) → ⑥ranked (tournoi) → ⑦monteur + export
```

Et le graphe de dépendances réel — `③` est en fait deux nœuds frères, pas une
étape séquentielle, et `⑦` (le monteur) ne s'intercale nulle part : il rejoue
`candidates` avec `--ban`/`--pin` en paramètres, il ne suit pas `ranked` :

```
rush ─┬─ probe ─────────────────────────────────┐
      │                                         │
      ├─ scenes ─┬─ thumbs ─── annot  (Haiku) ───┤
      │          │                              ├─ segments ─┬─────────────┐
      │          └─ metrics (OpenCV, réso native)│            │             │
      └──────────────────────────────────────────┘            ▼             │
                                                       candidates (CP-SAT)  │
                                                              │             │
                                                              └──► ranked ◄─┘
                                                                     │
                                                                     ├─ alternates
                                                                     ├─ render
                                                                     └─ exports ──► NLE (Resolve)
                                                                                     │
                                                                     conform ◄───────┘
```

(① = `scenes`, ② = `thumbs`, ③ = `metrics` + `annot`, ④ = `segments`,
⑤ = `candidates`, ⑥ = `ranked`, ⑦ = le monteur — `--ban`/`--pin`, puis export.)

1. **Découper chaque rush en plans** (`scenes`) — détection des coupures
   (filtre `scene` de ffmpeg), échantillonnées uniformément sur toute la durée
   si le rush dépasse `MAX_SEGMENTS_PER_RUSH` plans.
2. **Extraire une vignette par plan** (`thumbs`) — une image 640 px, la seule
   chose que le modèle de vision verra jamais.
3. **Mesurer et juger, en parallèle et sans se mélanger** (`metrics` + `annot`)
   — `metrics` calcule netteté, exposition, stabilité sur les pixels
   d'origine, sans aucun modèle ; `annot` envoie les vignettes à Haiku, qui
   juge l'intérêt du cadre, l'émotion, le rôle narratif. Les deux nœuds sont
   frères, pas l'un en aval de l'autre : un plan peut être narrativement
   indispensable et techniquement médiocre, ce n'est pas au modèle de
   trancher.
4. **Fusionner en une liste de plans exploitables** (`segments`) — c'est le
   seul nœud qui ne prend aucun paramètre de montage : il porte le résultat de
   la vision, le poste de coût, et rien ne doit pouvoir le réinvalider par
   accident. Un plan dont l'annotation a échoué est écarté, jamais maquillé en
   donnée valide.
5. **Résoudre le puzzle sous contraintes** (`candidates`) — CP-SAT choisit et
   ordonne les plans (durée cible, pas de répétition, pas deux plans de la
   même source à la suite, ouverture/clôture par rôle, positions imposées par
   `--pin`…), pour chacune des intentions du faisceau (`--presets`), K
   solutions distinctes par intention.
6. **Classer par tournoi** (`ranked`) — Sonnet compare deux candidats à la
   fois sur leurs raccords et désigne un gagnant, jusqu'à classer tout le
   faisceau. Ce nœud s'exécute toujours, y compris en `--rank manual` : c'est
   le mode qui supprime l'appel au modèle (le classement revient au monteur),
   pas un contournement du nœud — `alternates` dépend donc de `ranked`, et non
   de `candidates`.
7. **Le monteur tranche, puis ça sort** — veto (`--ban`) ou pin (`--pin`),
   rejeu ; le candidat retenu part en mp4 (`render`), EDL/FCPXML/plan JSON
   (`exports`), ou directement dans une timeline DaVinci Resolve ouverte
   (`--resolve`). `conform` n'est pas un nœud du graphe mais un outil séparé,
   qui relit ce que le NLE a produit et le retraduit en vetos.

Trois principes portent le reste.

**Le LLM annote, il n'assemble pas.** Sélectionner et ordonner sous contraintes
est un problème d'optimisation combinatoire, pas une tâche de langage. CP-SAT le
résout en un passage : il n'y a rien à « réviser », donc aucune boucle.

**Ce qui se mesure n'est pas deviné.** La netteté, l'exposition et la stabilité
se calculent sur les pixels d'origine. Le modèle garde ce qu'une vignette permet
réellement de juger : le cadre, le sujet, la valeur narrative.

**Le monteur est une entrée du graphe.** Son veto invalide le plan, jamais
l'annotation vision. Le cycle « je lance, je regarde, je bannis, je relance »
coûte quelques secondes de solveur et zéro token.

### Les agents

Deux, et deux seulement.

| Agent | Nœud | Modèle | Rôle |
|---|---|---|---|
| `AnnotatorAgent` | `annot` | Haiku 4.5, ou un VLM local | Tags sémantiques, émotion, rôle narratif, intérêt du plan |
| `ComparatorAgent` | `ranked` | Sonnet 4.6 | Comparaison par paires sur les frames de raccord |

Les six agents de l'implémentation d'origine — ANALYZER, SCENARIO, CRITIC,
REVISION, QUALITY, EDITOR — ne sont référencés que par `orchestrator.py` et
n'existent donc que sous `--engine loop`. Une seule dépendance croisée subsiste,
dans le bon sens : `analyzer.py` importe le contrat `SegmentSemantics` depuis
`annotator.py`.

### Les nœuds

| Nœud | Calcul | Coût |
|---|---|---|
| `probe` | métadonnées via imageio-ffmpeg | local, négligeable |
| `scenes` | détection de plans, échantillonnage uniforme sur la durée | local |
| `thumbs` | une vignette 640 px par plan | local |
| `metrics` | netteté, exposition, stabilité en pleine résolution | local, ~120 ms/plan en 720p |
| `annot` | Haiku : tags, émotion, rôle narratif, intérêt | **API**, lots de 4 images |
| `segments` | fusion, plans en échec écartés | local |
| `candidates` | CP-SAT : K solutions par intention, déduplication | local, quelques secondes |
| `ranked` | Sonnet : comparaison par paires sur les frames de raccord | **API**, ~10 appels |
| `alternates` | export EDL/FCPXML de tout le faisceau | local |
| `render` / `exports` | mp4, EDL, FCPXML, plan JSON du candidat retenu | local |

Deux nœuds seulement sortent sur le réseau. Le reste est local et reproductible.

### Coût & latence (`src/usage.py`)

`annot` et `ranked` sont les deux seuls nœuds facturés ; chaque appel de
`BaseAgent.call` chronomètre la requête et lit `response.usage`, dans un
accumulateur global réinitialisé au début de chaque `Run` — pas dans le
cache, où un chiffre gelé la première fois mentirait à chaque relecture. Le
rapport s'affiche en fin de run :

```
── Coût & latence (API, ce run) ────────────────────────
  AnnotatorAgent      14 appel(s)   38200 in /   6100 out tok    22.4s  $0.0687
  ComparatorAgent       9 appel(s)   12400 in /   2300 out tok    18.9s  $0.0717
  total                23 appel(s)   50600 in /   8400 out tok    41.3s  $0.1404
```

Les tarifs (`MODEL_PRICING_PER_MTOK` dans `src/config.py`) sont indicatifs :
l'API ne les expose pas, un modèle absent de la table compte pour $0 plutôt
que pour un prix deviné. Un nœud servi par le cache ne fait aucun appel, donc
n'apparaît pas dans ce rapport — c'est la même logique que la colonne « Coût »
ci-dessus.

### Le graphe (`src/graph.py`, `src/nodes.py`)

Les nœuds par rush sont indépendants entre rushes, et `metrics` est un frère de
`annot`, pas son successeur. Conséquences vérifiées par
`tests/test_pipeline_e2e.py` :

- ajouter un rush ne réanalyse que ce rush ;
- passer la durée cible de 60 à 90 s n'invalide que `candidates` et l'aval,
  jamais les annotations vision, qui sont le poste de coût ;
- un veto du monteur invalide le plan, pas l'analyse ;
- un plantage en aval ne perd pas le travail amont ;
- `--explain` affiche le graphe et ce qui serait recalculé.

Le corps des fonctions n'est **pas** haché : c'est `V[...]` dans `src/nodes.py`
qui déclare qu'un calcul a changé de sémantique, comme une migration. Oublier de
l'incrémenter fait servir une valeur périmée — c'est arrivé pendant le
développement, et c'est le piège principal de cette architecture.

Le cache impose aussi le déterminisme : un nœud non déterministe rend la valeur
relue différente d'un recalcul. C'est pourquoi le solveur tourne sur un worker
avec une limite en temps déterministe plutôt qu'en portefeuille multi-thread.

### Les mesures locales (`src/video/metrics.py`)

La vignette envoyée à l'API fait 640 px de large en JPEG : à cette échelle, un
rush 4K légèrement flou et un rush net sont identiques, et un demi-stop
d'écrêtage a disparu dans la recompression. On posait au modèle une question
dont l'information n'était plus dans l'image.

| Mesure | Comment | Piège évité |
|---|---|---|
| Netteté | rapport entre l'énergie haute fréquence de l'image et celle de sa version floutée, normalisé en log | la variance brute du Laplacien classe la *texture* : un mur de briques net bat un ciel net |
| Exposition | pixels aux extrémités de l'histogramme, hautes lumières pondérées plus fort que les noirs | les noirs bouchés sont souvent intentionnels et se rattrapent à l'étalonnage |
| Stabilité | transformation dominante entre images consécutives, par RANSAC sur points suivis | la corrélation de phase mesure le mouvement *apparent* : une caméra posée devant une piste de danse en ressortait « instable » |

`technical` est le **minimum** des trois, pas la moyenne : un plan cramé mais net
et stable, un monteur le jette. Une mesure impossible vaut `None`, jamais 1.0 —
le suivi de points échoue d'abord sur les plans très secoués, donc une valeur
neutre donnerait la meilleure note aux pires plans.

Les seuils (`_SHARP_FLOOR`, `_CLIP_TOLERANCE`, `_JITTER_TOLERANCE`) sont calés
sur mires de synthèse. Sur de vrais rushes — grain, optique ouverte, flou
d'arrière-plan volontaire — ils sont à revoir.

### Faire tourner la vision en local

Le dépôt ne dépend d'aucun runtime : il parle l'API compatible OpenAI, que
vLLM, Ollama, llama.cpp server, LM Studio et SGLang exposent tous. La pile CUDA
reste donc un problème d'installation du serveur, pas du projet.

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct --limit-mm-per-prompt image=4
python -m src.main rushes/ --annot-model local/Qwen/Qwen3-VL-8B-Instruct
python -m src.main rushes/ --annot-model local/qwen3-vl --local-url http://localhost:11434/v1
```

Le préfixe `local/` fait partie du nom du modèle, donc de la clé du nœud
`annot` : basculer cloud ↔ local invalide les annotations et **rien d'autre**.
`scenes`, `thumbs` et `metrics` sont conservés.

**Sortie structurée.** C'est le point qui décide si ça marche. Toute
l'architecture repose sur des contrats Pydantic ; sans décodage contraint, un
VLM local produit du JSON invalide assez souvent pour rendre le pipeline
inutilisable. Trois modes sont négociés puis mémorisés :
`response_format: json_schema`, `guided_json`, puis schéma dans le prompt avec
relance de réparation. Si aucun des deux premiers n'est accepté, l'erreur nomme
des serveurs qui conviennent plutôt que de laisser croire que le mode dégradé
suffit ; et un serveur qui prétend contraindre puis sort hors schéma lève, au
lieu d'être rattrapé — le masquer cacherait une mauvaise configuration.
Forçable avec `LOCAL_VLM_STRUCTURED_MODE`.

**Vignettes.** Un serveur local ne facture pas au token : la largeur passe
automatiquement de 640 à 1280 px (`--thumbnail-width` pour forcer). C'est un
gain réel — 640 px ne permettent pas de juger un cadre sur un rush 4K. Changer
la largeur invalide `thumbs` et `annot`, pas les mesures.

**Coût et latence.** Les appels locaux passent par le même `UsageTracker` que
les appels API : tokens quand le serveur les remonte, latence toujours, coût à
0 puisque le modèle est hors grille tarifaire. C'est ce qui permet de comparer
les deux sur autre chose que l'intuition.

**Sur une carte Blackwell 16 Go (RTX 5070 Ti, sm_120)**

Deux contraintes indépendantes.

*L'architecture.* sm_120 n'est pas dans toutes les roues prébuilt : selon les
versions, PyTorch, vLLM ou llama.cpp doivent être installés en build CUDA 12.8+
ou compilés avec `-DGGML_CUDA_ARCHITECTURES=120`. Symptôme typique :
`no kernel image is available for execution on the device`, ou un repli
silencieux sur CPU. Vérifier avant de brancher le pipeline —
`python -c "import torch; print(torch.cuda.get_device_capability())"` doit
renvoyer `(12, 0)`.

*La VRAM.* 16 Go tiennent confortablement un VLM de 7–8B quantifié
(≈ 6–9 Go de poids), mais le budget se joue ailleurs : l'encodeur visuel et le
cache KV grossissent avec le **nombre d'images par requête** et leur
résolution. `ANNOTATE_BATCH_SIZE = 4` à 1280 px est le poste dominant. En cas
d'OOM, réduire d'abord le lot à 2, puis la largeur, et seulement en dernier la
taille du modèle : l'annotation dépend moins du nombre de paramètres que de la
résolution.

Les familles utilisables dans ce budget évoluent vite ; le dépôt n'en code
aucune en dur, précisément pour cette raison.

**Mesurer plutôt que supposer**

```bash
python -m src.bench_annot rushes/ \
    --a claude-haiku-4-5-20251001 \
    --b local/Qwen/Qwen3-VL-8B-Instruct
```

Les deux runs partagent `scenes`, `thumbs` et `metrics` : la comparaison ne
coûte qu'une annotation. Le rapport donne l'accord champ à champ, la
corrélation de **rang** sur les notes d'intérêt — deux modèles n'utilisant pas
la même échelle, seul l'ordre compte pour le solveur — et surtout le Jaccard
sur la sélection finale. C'est ce dernier qui tranche : le solveur absorbe une
partie du bruit d'annotation, et un désaccord de 80 % sur l'émotion peut ne
rien changer au montage. Le classement est forcé en mode manuel, sans veto ni
pin, pour ne pas mélanger d'autres variances à celle qu'on mesure.

### Le solveur (`src/assemble.py`)

`x[i][p] = 1` si le segment `i` occupe la position `p`.

Contraintes dures : durée dans la bande de tolérance, chaque segment au plus une
fois, positions contiguës, pas deux plans adjacents du même rush, ouverture en
tête, résolution en queue, ordre chronologique optionnel.

Objectif : intérêt jugé par le modèle, note technique mesurée, adéquation
rôle/position, écart à une courbe d'énergie, écart au rythme cible, moins un
péage par plan. Les deux premiers termes sont **séparés** et non fondus en une
note unique : un plan peut être narrativement indispensable et techniquement
médiocre, c'est au preset d'arbitrer et non au modèle de moyenner à l'aveugle.
Sans le péage, une somme de scores positifs veut toujours plus de plans et sature
le plafond de durée.

Chaque contrainte est testable et se discute avec un monteur, contrairement à un
prompt de 400 mots. `Candidate.gap` expose l'écart à la borne supérieure : on
sait ce qu'on ne sait pas.

### Le faisceau (`src/beam.py`)

Cinq intentions — `chronological`, `emotional_arc`, `punchy`, `contemplative`,
`best_of` — sont des `Preset`, pas des prompts : générer le faisceau coûte zéro
token et les solveurs sont indépendants. Le solveur énumère K solutions
distinctes par intention (no-good de diversité), on déduplique au Jaccard, puis
on classe par comparaison par paires.

Pas de note absolue, donc pas de seuil arbitraire à justifier : un modèle n'a pas
besoin d'être calibré pour ordonner. Le comparateur reçoit les **frames de
raccord** (`src/video/cuts.py`) — la dernière image avant chaque coupe et la
première après — et juge donc le montage plutôt que le contenu.

### La boucle humaine

```bash
# 1. sortir le faisceau sans qu'aucun modèle ne tranche
python -m src.main rushes/ --rank manual
#    → K timelines en EDL/FCPXML, à comparer dans le NLE sur du mouvement
#      plutôt que sur des images fixes

# 2. choisir, et retirer les plans qu'on ne veut pas voir
python -m src.main rushes/ --pick 2 --ban rush_0@12.250,rush_1@3.000
python -m src.main rushes/ --ban-file bans.json

# 3. ou imposer un plan à une position (contrainte dure, pas une suggestion)
python -m src.main rushes/ --pin rush_0@12.250=0 --pin rush_2@8.400=last
python -m src.main rushes/ --pin-file pins.json
```

Les clés de veto et de pin sont celles qu'affiche le rapport d'assemblage et
que porte `source_key` dans le plan JSON exporté — pas un identifiant interne
à reconstituer. Une clé qui ne correspond à rien fait échouer le run : l'ignorer
rendrait un montage inchangé et laisserait croire au veto (ou au pin).

`--pin` prend `CLE=POSITION`, POSITION étant un entier 0-based ou `first`/
`last`. C'est une ligne de CP-SAT (`x[i][position] == 1`), donc garanti —
contrairement à un prompt qui ne ferait que suggérer. Un pin incompatible avec
les autres contraintes du preset (chronologie stricte, deux plans pinnés à la
même position, position hors bornes pour la durée cible) rend le candidat
infaisable : il disparaît du faisceau plutôt que de rendre un montage qui
l'ignore silencieusement.

`--rank manual` mérite d'être le défaut sur un vrai projet. « Lequel des deux tu
livrerais » est une question de monteur, et l'automatiser était le dernier
endroit où un modèle jouait encore le rôle de l'humain.

### Le retour depuis le NLE (`src/conform.py`)

```bash
python -m src.conform output/01_punchy_plan.json monté.fcpxml --write-bans bans.json
python -m src.main rushes/ --ban-file bans.json
```

On relit la timeline conformée et on la compare au plan proposé. L'appariement se
fait par recouvrement et non par égalité : un monteur rogne presque toujours, et
compter un plan raccourci comme « écarté puis ajouté » fausserait le résultat
dans les deux sens.

En sortie : les plans écartés deviennent des clés de veto, les plans ajoutés sont
signalés, et surtout un **taux d'accord**. C'est le premier signal d'évaluation
du dépôt qui ne soit pas l'avis d'un modèle sur son propre travail, et le seul
moyen que `quality_weight`, `technical_weight` et les tables d'énergie cessent
d'être des nombres choisis au jugé.

---

## Outputs

Les livrables atterrissent dans `output/` avec des noms lisibles ; le cache vit
sous `output/cache/`, indexé par clé.

| Fichier | Description |
|---|---|
| `montage_NN_<titre>.mp4` | Rendu du candidat retenu |
| `<titre>.edl` | EDL CMX 3600 — import DaVinci Resolve |
| `<titre>.fcpxml` | FCPXML 1.10 — import Resolve / Final Cut |
| `<titre>_plan.json` | Plan complet, entrée de `src.export_resolve` et de `src.conform` |
| `output/cache/` | Artefacts du graphe, adressés par contenu |
| `output/thumbnails/` | Frames extraites pour la vision et les raccords |

En `--rank manual`, tout le faisceau est exporté (`01_…`, `02_…`) sans rendu mp4.

### Importer dans DaVinci Resolve

**EDL** : `Fichier > Importer > Chronologie` → sélectionner le `.edl`
**FCPXML** : `Fichier > Importer > Chronologie` → sélectionner le `.fcpxml`

> Les deux formats référencent les fichiers sources originaux. DaVinci peut faire
> un conform à pleine qualité sans repasser par le mp4 rendu.

### Créer la timeline directement dans Resolve (API)

Sans passer par un import de fichier : le plan est construit dans le projet
Resolve courant via l'API de scripting (import des médias dans le Media Pool,
`CreateEmptyTimeline` + `AppendToTimeline`, in/out au fps réel de chaque clip,
bornes re-clampées aux limites du média — garde anti-hallucination du plan).

Prérequis : **Resolve Studio ouvert**, projet actif, et
`Préférences > Système > Général > External scripting using` = **Local**.

```bash
python -m src.main rushes/ --resolve
python -m src.export_resolve output/<titre>_plan.json
```

Hors console interne de Resolve, exposer le module de scripting :

```bash
# macOS
export RESOLVE_SCRIPT_API="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
# Windows (PowerShell)
$env:RESOLVE_SCRIPT_API="$env:PROGRAMDATA\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting"
```

Limites de l'API (signalées en console, jamais silencieuses) : les fondus et le
retime (`speed_factor ≠ 1`) ne peuvent pas être posés par `AppendToTimeline` —
ils restent à appliquer dans Resolve, et figurent par ailleurs dans l'EDL et le
FCPXML exportés.

---

## Structure du projet

```
montage-auto/
├── app.py                    # UI Streamlit (moteur historique uniquement)
├── requirements.txt
├── .env                      # ANTHROPIC_API_KEY (non versionné)
├── src/
│   ├── config.py             # Seuils, modèles, chemins
│   ├── models.py             # Contrats Pydantic + segment_key()
│   ├── main.py               # Entrypoint CLI, choix du moteur
│   │
│   ├── graph.py              # Moteur de graphe adressé par contenu
│   ├── nodes.py              # Définition des nœuds et de leurs versions
│   ├── pipeline.py           # Orchestration par le graphe + rapports
│   ├── assemble.py           # Modèle CP-SAT, presets, EditPlan
│   ├── beam.py               # Faisceau, déduplication, classement par paires
│   ├── conform.py            # Relecture d'une timeline montée, diff, vetos
│   ├── bench_annot.py        # Comparaison de deux modèles d'annotation
│   │
│   ├── orchestrator.py       # Machine à états historique (--engine loop)
│   ├── export.py             # Générateurs EDL + FCPXML
│   ├── export_resolve.py     # Timeline directe dans Resolve (scripting API)
│   │
│   ├── agents/
│   │   ├── base_agent.py     # Wrapper d'agent, délègue au backend
│   │   ├── backends.py       # Anthropic | endpoint compatible OpenAI
│   │   ├── annotator.py      # ANNOTATOR — alignement strict, échec explicite
│   │   ├── comparator.py     # COMPARATOR — comparaison par paires
│   │   ├── analyzer.py       # ANALYZER   ┐
│   │   ├── scenario.py       # SCENARIO   │
│   │   ├── critic.py         # CRITIC     ├ moteur historique
│   │   ├── revision.py       # REVISION   │
│   │   ├── quality.py        # QUALITY    │
│   │   └── editor.py         # EDITOR     ┘
│   └── video/
│       ├── probe.py          # Métadonnées + détection de scènes
│       ├── thumbnails.py     # Extraction de frames pour la vision
│       ├── metrics.py        # Netteté, exposition, stabilité (OpenCV)
│       ├── cuts.py           # Frames de part et d'autre de chaque coupe
│       └── editor.py         # Exécuteur moviepy
└── tests/                    # 85 tests, sans clé API ni GPU
```

---

## Configuration

Les paramètres principaux sont dans `src/config.py`. La plupart se surchargent en
ligne de commande.

**Moteur graphe**

| Paramètre | Défaut | Description |
|---|---|---|
| `TARGET_MONTAGE_DURATION` | `60` | Durée cible en secondes (`--duration`) |
| `MAX_SEGMENTS_PER_RUSH` | `40` | Plans retenus par fichier, échantillonnés uniformément sur toute la durée |
| `METRIC_SAMPLES` | `3` | Points de mesure locale par plan |
| `ANNOTATE_BATCH_SIZE` | `4` | Images par appel vision |
| `K_PER_PRESET` | `2` | Solutions distinctes demandées au solveur par intention |
| `SOLVER_TIME_LIMIT_S` | `15.0` | Budget CP-SAT par solution |
| `DEDUPE_THRESHOLD` | `0.85` | Jaccard au-delà duquel deux candidats sont redondants |
| `MAX_CANDIDATES` | `6` | Plafond de candidats soumis au classement |
| `COMPARATOR_MAX_CUTS` | `4` | Raccords montrés par candidat |
| `ANALYZER_MODEL` | Haiku 4.5 | Annotation, gros volume (`--annot-model`) |
| `LOCAL_VLM_BASE_URL` | `http://localhost:8000/v1` | Serveur compatible OpenAI (`--local-url`) |
| `LOCAL_VLM_STRUCTURED_MODE` | `auto` | `auto` \| `json_schema` \| `guided_json` \| `prompt` |
| `LOCAL_VLM_TIMEOUT_S` | `300` | Un VLM local est lent sur un lot d'images |
| `LOCAL_THUMBNAIL_WIDTH` | `1280` | Largeur des vignettes en mode local |
| `COMPARATOR_MODEL` | Sonnet 4.6 | Comparaison par paires |

Les poids de l'objectif (`quality_weight`, `technical_weight`, `role_weight`,
`energy_weight`, `pacing_weight`, `shot_cost`) sont définis par intention, dans
`PRESETS` de `src/assemble.py`.

**Moteur historique** : `QUALITY_THRESHOLD` (`0.70`), `MAX_ITERATIONS` (`3`),
`SCENARIO_MODEL`, `CRITIC_MODEL`, `REVISION_MODEL`, `QUALITY_MODEL`.

---

## Tests

```bash
pip install pytest && python -m pytest tests -q
```

85 tests, sans clé API ni GPU. Les tests de bout en bout emploient des doubles
déterministes pour les deux agents LLM et des fixtures vidéo générées par ffmpeg
(non versionnées ; script de régénération dans `tests/fixtures/README.md`). Ils
se skippent proprement si le dossier est vide.

---

## Le moteur historique (`--engine loop`)

L'implémentation d'origine : une machine à états écrite à la main, avec une
boucle de révision. Conservée pour comparaison.

```
ANALYZER → SCENARIO → EDITOR → CRITIC ─┬─ score ≥ 0.70 → QUALITY → sortie
                          ▲            │
                       REVISION ◄──────┘ score < 0.70 et iter < max
```

```bash
python -m src.main rushes/ --engine loop --max-iter 3
```

Ce qui a motivé son remplacement :

| | `--engine loop` | `--engine graph` |
|---|---|---|
| Ordonnancement | ordre écrit à la main | graphe de dépendances, remonté à la demande |
| Assemblage | SCENARIO (LLM) écrit la timeline | CP-SAT sous contraintes déclarées |
| Exploration | boucle séquentielle, faisceau de 1 | N intentions indépendantes |
| Évaluation | score absolu contre un seuil de 0.70 | comparaison par paires, puis taux d'accord avec le montage livré |
| Netteté, expo, stabilité | devinées par le modèle sur 640 px | mesurées en pleine résolution |
| Sortie | un montage | un classement : livré + alternates |
| Humain | absent, simulé par le CRITIC | veto, choix, arbitrage sans modèle |
| Reprise | aucune | propriété du cache |
| Rejouabilité | non | oui, solveur déterministe |

La boucle était une recherche dégradée : un seul candidat exploré à la fois, une
récompense scalaire produite par un modèle jugeant ses propres propositions,
aucune garantie de monotonie, un rendu vidéo complet par itération pour n'être
noté que sur trois images, et rien de réutilisable d'un run à l'autre.

---

## Limites

- **Pas d'audio du tout.** `has_audio` est relevé et jamais lu. C'est le manque
  le plus visible pour un monteur, le rythme d'un montage étant piloté par la
  parole et la musique. La suite naturelle est d'en faire des contraintes du
  solveur (« chaque coupe à ±80 ms d'un onset »), ce qu'un prompt ne sait pas
  garantir et qu'un modèle CP-SAT garantit par construction.
- **Le comparateur reste dans le cloud.** L'annotation peut tourner en local
  (`--annot-model local/…`), mais le classement par paires n'a pas d'équivalent
  local testé : il demande un raisonnement sur plusieurs images à la fois. En
  attendant, `--rank manual` supprime cet appel — et rend le pipeline
  entièrement local si l'annotation l'est aussi.
- **L'écart local/cloud n'est pas chiffré.** `src.bench_annot` existe pour le
  mesurer, mais aucun chiffre n'a été produit sur de vrais rushes.
- **Seuils des métriques non calibrés** sur de vrais rushes (cf. plus haut).
- Le comparateur juge des images fixes de raccord, pas le mouvement : un faux
  raccord sur un travelling lui échappe encore.
- Aucune notion de multicam : ni synchronisation, ni choix d'angle sur action
  simultanée.
- La détection de scènes repose sur le filtre `scene` de ffmpeg (différence
  d'histogramme), pas sur un modèle de vision dédié : elle rate les coupes
  franches entre plans visuellement proches.
