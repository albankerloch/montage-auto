"""Assemblage de la timeline par programmation par contraintes.

Le LLM n'écrit plus la timeline. Il annote (`quality_score`, rôle, émotion) ;
le solveur sélectionne et ordonne sous contraintes explicites.

Ce que ça change concrètement :
  - l'optimum sous contraintes est atteint en un passage : il n'y a plus rien à
    « réviser », donc plus de boucle ;
  - le solveur énumère les K meilleures solutions distinctes, donc les
    « alternates » sont gratuits — une boucle ne rend qu'un verdict ;
  - les contraintes (« pas deux plans de la même source à la suite », « durée
    entre 55 et 65 s ») se discutent avec un monteur, contrairement à un prompt.

Formulation : x[i][p] = 1 si le segment i occupe la position p.
"""
from __future__ import annotations

import math
from typing import Literal, Sequence

from ortools.sat.python import cp_model
from pydantic import BaseModel, Field

from src.models import EditPlan, TimelineEdit, VideoSegment, segment_key

# ── Échelles sémantiques ────────────────────────────────────────────────────
# Le LLM produit des étiquettes ; ces tables les projettent sur [0,1] pour que
# le solveur puisse arbitrer. Elles sont volontairement dans le code et pas dans
# un prompt : ce sont des paramètres à régler, pas du langage à interpréter.

ENERGY: dict[str, float] = {
    "energetic": 1.00,
    "joyful": 0.85,
    "tense": 0.70,
    "suspenseful": 0.65,
    "neutral": 0.50,
    "calm": 0.25,
    "melancholic": 0.15,
}

ROLE_POSITION: dict[str, float] = {
    "opening": 0.00,
    "build_up": 0.35,
    "climax": 0.65,
    "resolution": 0.85,
    "outro": 1.00,
    "b_roll": 0.50,
}

CurveName = Literal["arc", "rise", "fall", "flat"]


def energy_curve(name: CurveName, pos: float) -> float:
    """Énergie visée à la position relative `pos` ∈ [0,1]."""
    if name == "flat":
        return 0.50
    if name == "rise":
        return 0.20 + 0.80 * pos
    if name == "fall":
        return 1.00 - 0.80 * pos
    # "arc" : montée jusqu'au pic à 70 %, puis résolution
    if pos <= 0.70:
        return 0.20 + (1.00 - 0.20) * (pos / 0.70)
    return 1.00 - (1.00 - 0.35) * ((pos - 0.70) / 0.30)


# ── Préréglages = les « intentions » du faisceau ────────────────────────────


class Preset(BaseModel):
    """Une intention de montage, entièrement déclarative.

    Ces profils sont déterministes : générer le faisceau ne coûte aucun token.
    Le LLM n'intervient qu'au classement (cf. `src/beam.py`).
    """

    name: str
    target_duration: float = 60.0
    tolerance: float = 0.10
    min_shot: float = 1.2
    max_shot: float = 6.0
    target_shot: float = 3.0

    quality_weight: float = 1.0     # intérêt jugé par le modèle
    technical_weight: float = 1.0   # défauts mesurés localement
    role_weight: float = 0.6
    energy_weight: float = 0.6
    pacing_weight: float = 0.4

    curve: CurveName = "arc"
    respect_chronology: bool = False
    forbid_same_source_adjacent: bool = True
    require_opening_first: bool = True
    require_closing_last: bool = True
    min_shots: int = 4
    shot_cost: float = 0.80  # péage par plan : un plan médiocre ne s'ajoute pas gratuitement

    default_transition: str = "cut"
    open_with_fade: bool = True
    dissolve_on_emotion_change: bool = False


PRESETS: dict[str, Preset] = {
    "chronological": Preset(
        name="chronological",
        curve="rise",
        respect_chronology=True,
        target_shot=3.5,
        quality_weight=1.0,
        role_weight=0.2,
        energy_weight=0.3,
        forbid_same_source_adjacent=False,  # incompatible avec un ordre chronologique global
        require_opening_first=False,
        require_closing_last=False,
    ),
    "emotional_arc": Preset(
        name="emotional_arc",
        curve="arc",
        energy_weight=1.2,
        role_weight=0.9,
        target_shot=3.5,
        dissolve_on_emotion_change=True,
    ),
    "punchy": Preset(
        name="punchy",
        curve="rise",
        min_shot=0.8,
        max_shot=2.5,
        target_shot=1.4,
        pacing_weight=1.0,
        energy_weight=0.9,
        role_weight=0.3,
        shot_cost=0.25,
    ),
    "contemplative": Preset(
        name="contemplative",
        curve="fall",
        min_shot=2.5,
        max_shot=9.0,
        target_shot=5.5,
        pacing_weight=0.8,
        energy_weight=0.5,
        shot_cost=1.40,
        dissolve_on_emotion_change=True,
    ),
    "best_of": Preset(
        name="best_of",
        curve="flat",
        quality_weight=2.0,
        technical_weight=2.0,
        role_weight=0.0,
        energy_weight=0.0,
        pacing_weight=0.2,
        require_opening_first=False,
        require_closing_last=False,
    ),
}

DEFAULT_BEAM = ["chronological", "emotional_arc", "punchy", "contemplative", "best_of"]


# ── Sortie du solveur ───────────────────────────────────────────────────────


class Pick(BaseModel):
    index: int
    start: float
    end: float


class Candidate(BaseModel):
    preset: str
    rank_in_preset: int
    objective: int
    bound: int
    status: str
    picks: list[Pick]

    @property
    def gap(self) -> float:
        """Écart relatif à la borne supérieure. 0.0 = optimum prouvé."""
        if self.bound <= 0:
            return 0.0
        return max(0.0, (self.bound - self.objective) / self.bound)

    @property
    def duration(self) -> float:
        return sum(p.end - p.start for p in self.picks)

    @property
    def selection(self) -> frozenset[int]:
        return frozenset(p.index for p in self.picks)


def _trim(seg: VideoSegment, preset: Preset) -> tuple[float, float]:
    """Ramène un segment dans [min_shot, max_shot], centré sur son milieu."""
    dur = max(0.0, seg.end_time - seg.start_time)
    use = min(dur, preset.max_shot)
    offset = (dur - use) / 2.0
    start = seg.start_time + offset
    return start, start + use


def _segment_score(seg: VideoSegment, use_dur: float, pos: float, preset: Preset) -> float:
    role = ROLE_POSITION.get((seg.suggested_role or "").lower(), 0.5)
    energy = ENERGY.get((seg.emotion or "").lower(), 0.5)
    want = energy_curve(preset.curve, pos)

    s = preset.quality_weight * seg.quality_score
    if seg.technical_score is not None:
        # Terme séparé, et non fondu dans une note unique : un plan peut être
        # narrativement indispensable et techniquement médiocre. C'est au
        # preset d'arbitrer, pas au modèle de moyenner les deux à l'aveugle.
        s += preset.technical_weight * seg.technical_score
    s += preset.role_weight * (1.0 - abs(role - pos))
    s += preset.energy_weight * (1.0 - abs(energy - want))
    s -= preset.pacing_weight * min(
        1.0, abs(use_dur - preset.target_shot) / max(0.5, preset.target_shot)
    )
    return s


def solve(
    segments: Sequence[VideoSegment],
    preset: Preset,
    k: int = 2,
    min_diff: int = 3,
    time_limit_s: float = 15.0,
    deterministic: bool = True,
    excluded: frozenset[str] = frozenset(),
) -> list[Candidate]:
    """Retourne jusqu'à `k` timelines distinctes, classées par objectif.

    `excluded` retire des segments par clé, jamais en filtrant la liste passée :
    `Pick.index` est une position dans `segments`, donc retirer un élément en
    amont décalerait toutes les positions suivantes et les candidats
    pointeraient vers d'autres plans. C'est le même piège que l'appariement par
    index de l'ancien ANALYZER, et il s'était refermé pareil.

    `deterministic=True` n'est pas un confort : le cache de `src/graph.py` est
    adressé par contenu, donc une fonction non déterministe le rend menteur —
    la valeur relue diffère de la valeur qu'un recalcul produirait. CP-SAT en
    portefeuille multi-thread avec limite d'horloge n'est pas reproductible ;
    on passe donc à un worker et à une limite en temps *déterministe* (unités
    de travail, pas de secondes), au prix de la vitesse.
    """
    usable = [
        (i, s, *_trim(s, preset))
        for i, s in enumerate(segments)
        if (s.end_time - s.start_time) >= preset.min_shot
        and segment_key(s.source_file, s.start_time) not in excluded
    ]
    if len(usable) < preset.min_shots:
        return []

    idx = [u[0] for u in usable]
    segs = [u[1] for u in usable]
    starts = [u[2] for u in usable]
    ends = [u[3] for u in usable]
    durs_ms = [int(round((e - s) * 1000)) for s, e in zip(starts, ends)]
    n = len(segs)

    lo_ms = int(preset.target_duration * (1 - preset.tolerance) * 1000)
    hi_ms = int(preset.target_duration * (1 + preset.tolerance) * 1000)

    n_pos = min(n, max(preset.min_shots, math.ceil(hi_ms / 1000 / preset.min_shot)))

    model = cp_model.CpModel()
    x = [[model.NewBoolVar(f"x_{i}_{p}") for p in range(n_pos)] for i in range(n)]
    used = [model.NewBoolVar(f"used_{p}") for p in range(n_pos)]

    for i in range(n):
        model.AddAtMostOne(x[i])
    for p in range(n_pos):
        model.Add(sum(x[i][p] for i in range(n)) == used[p])
    # Les positions occupées forment un préfixe : pas de trou dans la timeline.
    for p in range(n_pos - 1):
        model.Add(used[p] >= used[p + 1])

    model.Add(sum(durs_ms[i] * x[i][p] for i in range(n) for p in range(n_pos)) >= lo_ms)
    model.Add(sum(durs_ms[i] * x[i][p] for i in range(n) for p in range(n_pos)) <= hi_ms)
    model.Add(sum(used) >= preset.min_shots)

    if preset.forbid_same_source_adjacent:
        by_source: dict[str, list[int]] = {}
        for i, s in enumerate(segs):
            by_source.setdefault(s.source_file, []).append(i)
        for group in by_source.values():
            if len(group) < 2:
                continue
            for p in range(n_pos - 1):
                model.Add(sum(x[i][p] + x[i][p + 1] for i in group) <= 1)

    if preset.respect_chronology:
        rank = {i: r for r, i in enumerate(sorted(range(n), key=lambda j: (segs[j].source_file, segs[j].start_time)))}
        order = []
        for p in range(n_pos):
            o = model.NewIntVar(0, n, f"o_{p}")
            model.Add(o == sum((rank[i] + 1) * x[i][p] for i in range(n)))
            order.append(o)
        for p in range(n_pos - 1):
            model.Add(order[p] < order[p + 1]).OnlyEnforceIf(used[p + 1])

    if preset.require_opening_first:
        openers = [i for i, s in enumerate(segs) if (s.suggested_role or "").lower() == "opening"]
        if openers:
            model.Add(sum(x[i][0] for i in openers) == 1)

    if preset.require_closing_last:
        closers = [
            i
            for i, s in enumerate(segs)
            if (s.suggested_role or "").lower() in ("resolution", "outro")
        ]
        if closers:
            for p in range(n_pos):
                is_last = model.NewBoolVar(f"last_{p}")
                if p == n_pos - 1:
                    model.Add(is_last == used[p])
                else:
                    model.AddBoolAnd([used[p], used[p + 1].Not()]).OnlyEnforceIf(is_last)
                    model.AddBoolOr([used[p].Not(), used[p + 1]]).OnlyEnforceIf(is_last.Not())
                model.Add(sum(x[i][p] for i in closers) == 1).OnlyEnforceIf(is_last)

    denom = max(1, n_pos - 1)
    coeff = [
        [
            int(round(1000 * _segment_score(segs[i], durs_ms[i] / 1000.0, p / denom, preset)))
            for p in range(n_pos)
        ]
        for i in range(n)
    ]
    shot_cost = int(round(1000 * preset.shot_cost))
    model.Maximize(
        sum(coeff[i][p] * x[i][p] for i in range(n) for p in range(n_pos))
        - shot_cost * sum(used)
    )

    chosen = [model.NewBoolVar(f"chosen_{i}") for i in range(n)]
    for i in range(n):
        model.Add(chosen[i] == sum(x[i][p] for p in range(n_pos)))

    out: list[Candidate] = []
    for rank_i in range(k):
        solver = cp_model.CpSolver()
        if deterministic:
            solver.parameters.num_workers = 1
            solver.parameters.max_deterministic_time = time_limit_s
            solver.parameters.max_time_in_seconds = time_limit_s * 4  # garde-fou
        else:
            solver.parameters.num_workers = 8
            solver.parameters.max_time_in_seconds = time_limit_s
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break

        picks: list[Pick] = []
        for p in range(n_pos):
            for i in range(n):
                if solver.Value(x[i][p]):
                    picks.append(Pick(index=idx[i], start=starts[i], end=ends[i]))
                    break
        out.append(
            Candidate(
                preset=preset.name,
                rank_in_preset=rank_i,
                objective=int(solver.ObjectiveValue()),
                bound=int(solver.BestObjectiveBound()),
                status=solver.StatusName(status),
                picks=picks,
            )
        )

        # No-good de diversité : la solution suivante doit différer d'au moins
        # `min_diff` segments — sinon on obtient K permutations du même montage.
        sel = [i for i in range(n) if solver.Value(chosen[i])]
        if not sel:
            break
        model.Add(sum(chosen[i] for i in sel) <= len(sel) - min_diff)

    out.sort(key=lambda c: -c.objective)
    for r, c in enumerate(out):
        c.rank_in_preset = r
    return out


def to_edit_plan(
    candidate: Candidate,
    segments: Sequence[VideoSegment],
    preset: Preset,
    title: str | None = None,
) -> EditPlan:
    """Convertit une solution du solveur en `EditPlan` (contrat existant)."""
    edits: list[TimelineEdit] = []
    prev_emotion: str | None = None

    for order, pick in enumerate(candidate.picks):
        base = segments[pick.index]
        seg = base.model_copy(
            update={
                "start_time": pick.start,
                "end_time": pick.end,
                "duration": pick.end - pick.start,
                "source_key": segment_key(base.source_file, base.start_time),
            }
        )

        if order == 0 and preset.open_with_fade:
            transition, tdur = "fade", 0.5
        elif (
            preset.dissolve_on_emotion_change
            and prev_emotion is not None
            and seg.emotion != prev_emotion
        ):
            transition, tdur = "dissolve", 0.4
        else:
            transition, tdur = preset.default_transition, 0.0

        edits.append(
            TimelineEdit(
                order=order,
                segment=seg,
                transition_in=transition,
                transition_duration=tdur,
                audio_level=1.0,
                speed_factor=1.0,
            )
        )
        prev_emotion = seg.emotion

    total = sum(e.segment.duration for e in edits)
    return EditPlan(
        title=title or f"montage_{preset.name}_{candidate.rank_in_preset}",
        total_duration=total,
        narrative_arc=(
            f"preset={preset.name} · courbe={preset.curve} · "
            f"{len(edits)} plans · objectif={candidate.objective}"
        ),
        edits=edits,
        music_suggestion=None,
    )


def explain(candidate: Candidate, segments: Sequence[VideoSegment], preset: Preset) -> str:
    """Trace lisible : pourquoi chaque plan est là. Auditable par un monteur."""
    lines = [
        f"[{preset.name}] rang {candidate.rank_in_preset} · objectif {candidate.objective} "
        f"· {candidate.duration:.1f}s · {len(candidate.picks)} plans "
        f"({candidate.status}, écart {candidate.gap:.1%})"
    ]
    denom = max(1, len(candidate.picks) - 1)
    for p, pick in enumerate(candidate.picks):
        s = segments[pick.index]
        pos = p / denom
        lines.append(
            f"  {p:>2}. {segment_key(s.source_file, s.start_time):<24} "
            f"[{pick.start:.2f}→{pick.end:.2f}] ({pick.end - pick.start:.1f}s) "
            f"i={s.quality_score:.2f}"
            + (
                f" t={s.technical_score:.2f}"
                f"(n{s.sharpness:.2f}/e{s.exposure:.2f}/"
                + (f"s{s.stability:.2f})" if s.stability is not None else "s—)")
                if s.technical_score is not None else " t=—"
            )
            + f" {s.suggested_role}/{s.emotion} "
            f"| énergie visée {energy_curve(preset.curve, pos):.2f} "
            f"| score {_segment_score(s, pick.end - pick.start, pos, preset):.3f}"
        )
    return "\n".join(lines)
