"""Le faisceau et le classement, avec un comparateur factice (zéro appel API)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import beam  # noqa: E402
from src.assemble import Candidate, Pick  # noqa: E402


def cand(preset: str, rank: int, obj: int, indices: list[int]) -> Candidate:
    return Candidate(
        preset=preset,
        rank_in_preset=rank,
        objective=obj,
        bound=obj,
        status="OPTIMAL",
        picks=[Pick(index=i, start=0.0, end=2.0) for i in indices],
    )


def test_dedupe_drops_near_identical():
    a = cand("p1", 0, 100, [1, 2, 3, 4, 5])
    b = cand("p2", 0, 90, [1, 2, 3, 4, 6])  # Jaccard 4/6 = 0.67
    c = cand("p3", 0, 80, [1, 2, 3, 4, 5])  # identique à a
    kept = beam.dedupe([a, b, c], threshold=0.85)
    assert len(kept) == 2 and kept[0] is a and kept[1] is b


def test_prefilter_never_compares_objectives_across_presets():
    """Deux presets = deux fonctions objectif : leurs valeurs ne sont pas
    comparables. Le pré-filtre doit garder le meilleur *par* preset."""
    weak_preset_best = cand("contemplative", 0, 5, [1, 2, 3])
    strong_preset_worse = cand("punchy", 1, 900, [4, 5, 6])
    strong_preset_best = cand("punchy", 0, 1000, [7, 8, 9])
    kept = beam.objective_prefilter(
        [weak_preset_best, strong_preset_worse, strong_preset_best], per_preset=1
    )
    presets = {c.preset for c in kept}
    assert presets == {"contemplative", "punchy"}
    assert weak_preset_best in kept, "un preset à faible objectif ne doit pas être éliminé"
    assert strong_preset_worse not in kept


def test_rank_orders_by_fake_comparator():
    """Comparateur factice : le candidat au plus grand nombre de picks gagne."""
    cands = [cand(f"p{i}", 0, 0, list(range(i + 2))) for i in range(5)]
    calls = {"n": 0}

    def cmp(a: Candidate, b: Candidate) -> tuple[str, str]:
        calls["n"] += 1
        return ("A" if len(a.picks) > len(b.picks) else "B", "factice")

    ranked, n = beam.rank(cands, cmp)
    assert [len(r.candidate.picks) for r in ranked] == [6, 5, 4, 3, 2]
    assert n == calls["n"] and n <= 12, f"trop de comparaisons: {n}"


def test_rank_memoizes_pairs():
    cands = [cand(f"p{i}", 0, 0, [i]) for i in range(4)]
    seen: list[tuple[int, int]] = []

    def cmp(a: Candidate, b: Candidate) -> tuple[str, str]:
        seen.append((a.picks[0].index, b.picks[0].index))
        return ("A", "")

    beam.rank(cands, cmp)
    norm = {tuple(sorted(p)) for p in seen}
    assert len(norm) == len(seen), "une paire a été comparée deux fois"


def test_orientation_is_stable_and_mixed():
    """L'ordre de présentation est déterministe (rejouable) mais ne s'aligne
    pas sur un preset — sinon le biais de position du LLM le favorise."""
    pool = [cand(f"preset_{i}", 0, 0, [i]) for i in range(12)]
    orientations = [beam._orientation(pool[0], p) for p in pool[1:]]
    assert beam._orientation(pool[0], pool[1]) is beam._orientation(pool[0], pool[1])
    assert len(set(orientations)) == 2, "orientation constante = biais systématique"


def test_rank_handles_single_candidate():
    ranked, n = beam.rank([cand("p", 0, 0, [1])], lambda a, b: ("A", ""))
    assert len(ranked) == 1 and n == 0


def test_rank_caps_candidate_count():
    cands = [cand(f"p{i}", 0, 0, [i]) for i in range(20)]
    ranked, _ = beam.rank(cands, lambda a, b: ("A", ""), max_candidates=6)
    assert len(ranked) == 6


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
