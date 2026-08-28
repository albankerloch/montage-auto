"""Le solveur : les contraintes déclarées sont réellement respectées.

C'est l'intérêt principal du passage à CP-SAT : ce sont des propriétés
testables, contrairement à « le prompt demande de varier les durées ».
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assemble import (  # noqa: E402
    DEFAULT_BEAM,
    PRESETS,
    Preset,
    solve,
    to_edit_plan,
)
from src.models import VideoSegment  # noqa: E402

ROLES = ["opening", "build_up", "climax", "resolution", "outro", "b_roll"]
EMOTIONS = ["energetic", "joyful", "tense", "neutral", "calm", "melancholic"]

# Budget court : ces tests vérifient des invariants, pas l'optimalité.
BUDGET = 3.0


def fixture(n_files: int = 4, per_file: int = 18, seed: int = 7) -> list[VideoSegment]:
    rng = random.Random(seed)
    segs: list[VideoSegment] = []
    for f in range(n_files):
        t = 0.0
        for _ in range(per_file):
            d = rng.uniform(1.0, 9.0)
            segs.append(
                VideoSegment(
                    source_file=f"rush_{f}.mp4",
                    start_time=t,
                    end_time=t + d,
                    duration=d,
                    quality_score=round(rng.uniform(0.2, 0.95), 2),
                    semantic_tags=["b_roll"],
                    emotion=rng.choice(EMOTIONS),
                    suggested_role=rng.choice(ROLES),
                )
            )
            t += d + rng.uniform(0.2, 2.0)
    return segs


SEGS = fixture()


def test_duration_within_tolerance():
    for name in DEFAULT_BEAM:
        p = PRESETS[name]
        for c in solve(SEGS, p, k=1, time_limit_s=BUDGET):
            lo = p.target_duration * (1 - p.tolerance)
            hi = p.target_duration * (1 + p.tolerance)
            assert lo <= c.duration <= hi, f"{name}: {c.duration:.1f}s hors [{lo}, {hi}]"


def test_no_segment_used_twice():
    for name in DEFAULT_BEAM:
        for c in solve(SEGS, PRESETS[name], k=1, time_limit_s=BUDGET):
            idx = [p.index for p in c.picks]
            assert len(idx) == len(set(idx)), f"{name}: segment répété"


def test_no_same_source_adjacent():
    p = PRESETS["emotional_arc"]
    assert p.forbid_same_source_adjacent
    for c in solve(SEGS, p, k=1, time_limit_s=BUDGET):
        srcs = [SEGS[pick.index].source_file for pick in c.picks]
        assert all(a != b for a, b in zip(srcs, srcs[1:])), "deux plans de suite du même rush"


def test_shot_length_bounds():
    for name in DEFAULT_BEAM:
        p = PRESETS[name]
        for c in solve(SEGS, p, k=1, time_limit_s=BUDGET):
            for pick in c.picks:
                d = pick.end - pick.start
                assert d <= p.max_shot + 1e-6, f"{name}: plan de {d:.2f}s > max {p.max_shot}"
                assert d >= p.min_shot - 1e-6, f"{name}: plan de {d:.2f}s < min {p.min_shot}"


def test_chronology_preset_is_chronological():
    p = PRESETS["chronological"]
    # L'instance la plus dure du faisceau : l'ordre strict couple toutes les
    # positions entre elles, là où les autres presets se décomposent bien.
    cands = solve(SEGS, p, k=1, time_limit_s=12.0)
    assert cands, "le preset chronologique doit produire une solution"
    for c in cands:
        keys = [(SEGS[pick.index].source_file, SEGS[pick.index].start_time) for pick in c.picks]
        assert keys == sorted(keys), "l'ordre chronologique n'est pas respecté"


def test_opening_first_and_closing_last():
    p = PRESETS["emotional_arc"]
    for c in solve(SEGS, p, k=1, time_limit_s=BUDGET):
        assert SEGS[c.picks[0].index].suggested_role == "opening"
        assert SEGS[c.picks[-1].index].suggested_role in ("resolution", "outro")


def test_k_best_are_distinct():
    cands = solve(SEGS, PRESETS["emotional_arc"], k=3, min_diff=3, time_limit_s=BUDGET)
    assert len(cands) >= 2, "le solveur doit énumérer plusieurs solutions"
    sels = [c.selection for c in cands]
    for i in range(len(sels)):
        for j in range(i + 1, len(sels)):
            assert len(sels[i] ^ sels[j]) >= 3, "solutions trop proches (no-good inopérant)"


def test_ranked_by_objective():
    cands = solve(SEGS, PRESETS["punchy"], k=3, time_limit_s=BUDGET)
    objs = [c.objective for c in cands]
    assert objs == sorted(objs, reverse=True)
    assert [c.rank_in_preset for c in cands] == list(range(len(cands)))


def test_deterministic():
    """Même entrée, même sortie : rejouable, contrairement à un plan de LLM."""
    a = solve(SEGS, PRESETS["contemplative"], k=1, time_limit_s=BUDGET)
    b = solve(SEGS, PRESETS["contemplative"], k=1, time_limit_s=BUDGET)
    assert [p.index for p in a[0].picks] == [p.index for p in b[0].picks]


def test_to_edit_plan_roundtrip():
    p = PRESETS["emotional_arc"]
    c = solve(SEGS, p, k=1, time_limit_s=BUDGET)[0]
    plan = to_edit_plan(c, SEGS, p)
    assert [e.order for e in plan.edits] == list(range(len(plan.edits)))
    assert abs(plan.total_duration - c.duration) < 1e-6
    assert plan.edits[0].transition_in == "fade"
    # Les bornes du plan sont dans les bornes réelles des segments source.
    for e in plan.edits:
        src = next(s for s in SEGS if s.source_file == e.segment.source_file
                   and s.start_time <= e.segment.start_time + 1e-6
                   and s.end_time >= e.segment.end_time - 1e-6)
        assert src is not None


def test_infeasible_returns_empty_not_garbage():
    tight = Preset(name="impossible", target_duration=600, tolerance=0.0, min_shot=1.0, max_shot=2.0)
    assert solve(SEGS[:5], tight, k=1, time_limit_s=5) == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
