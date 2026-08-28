"""La boucle humaine : veto, arbitrage manuel, choix d'un candidat.

Les deux premiers tests couvrent des bugs réels trouvés en exerçant la
fonctionnalité à la main :

  - le veto filtrait la liste passée au solveur, alors que `Pick.index` est une
    position dans cette liste : bannir un plan décalait tous les suivants et les
    candidats désignaient d'autres plans que ceux affichés. Même piège que
    l'appariement par index de l'ancien ANALYZER ;
  - la clé de veto affichée dans le rapport ne correspondait pas à la clé
    attendue par le filtre (nom de base contre chemin, 2 décimales contre 3,
    borne recadrée contre borne d'origine), donc aucun veto ne pouvait matcher.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assemble import PRESETS, solve  # noqa: E402
from src.models import segment_key  # noqa: E402
from tests.test_assemble import SEGS  # noqa: E402


def keys_of(candidate, segments) -> list[str]:
    return [segment_key(segments[p.index].source_file, segments[p.index].start_time)
            for p in candidate.picks]


def test_excluded_segments_never_appear():
    preset = PRESETS["punchy"]
    baseline = solve(SEGS, preset, k=1, time_limit_s=5.0)[0]
    banned = frozenset(keys_of(baseline, SEGS)[:3])

    for c in solve(SEGS, preset, k=1, time_limit_s=5.0, excluded=banned):
        assert not (set(keys_of(c, SEGS)) & banned), "un plan banni est ressorti"


def test_exclusion_does_not_shift_indices():
    """Le test de régression du bug : la clé affichée et les bornes du plan
    doivent désigner le même segment, veto ou pas."""
    preset = PRESETS["punchy"]
    banned = frozenset({segment_key(SEGS[0].source_file, SEGS[0].start_time)})

    for c in solve(SEGS, preset, k=1, time_limit_s=5.0, excluded=banned):
        for pick in c.picks:
            src = SEGS[pick.index]
            # Les bornes recadrées sont forcément à l'intérieur du segment que
            # `index` désigne. Un décalage d'indices casse cet invariant.
            assert src.start_time - 1e-6 <= pick.start
            assert pick.end <= src.end_time + 1e-6


def test_exclusion_changes_the_result():
    preset = PRESETS["punchy"]
    baseline = solve(SEGS, preset, k=1, time_limit_s=5.0)[0]
    banned = frozenset(keys_of(baseline, SEGS)[:3])
    after = solve(SEGS, preset, k=1, time_limit_s=5.0, excluded=banned)[0]
    assert keys_of(after, SEGS) != keys_of(baseline, SEGS)


def test_unknown_ban_key_is_rejected():
    """Une clé qui ne matche rien est une faute de frappe. L'ignorer rendrait un
    montage inchangé et laisserait croire au veto."""
    from src import nodes
    from src.models import AnalysisResult

    analysis = AnalysisResult(
        segments=list(SEGS[:8]),
        total_rushes_duration=100.0,
        recommended_output_duration=60.0,
        summary="",
    ).model_dump(mode="json")

    with pytest.raises(ValueError, match="inconnue"):
        nodes._candidates(
            analysis=analysis,
            presets=[PRESETS["punchy"].model_dump()],
            k_per_preset=1,
            time_limit_s=3.0,
            banned=["rush_0@9.0"],  # 1 décimale au lieu de 3
            dedupe_threshold=0.85,
        )


def test_unknown_pin_key_is_rejected():
    """Même logique que le veto : une clé de pin qui ne matche rien est une
    faute de frappe, pas une intention à ignorer silencieusement."""
    from src import nodes
    from src.models import AnalysisResult

    analysis = AnalysisResult(
        segments=list(SEGS[:8]),
        total_rushes_duration=100.0,
        recommended_output_duration=60.0,
        summary="",
    ).model_dump(mode="json")

    with pytest.raises(ValueError, match="inconnue"):
        nodes._candidates(
            analysis=analysis,
            presets=[PRESETS["punchy"].model_dump()],
            k_per_preset=1,
            time_limit_s=3.0,
            banned=[],
            dedupe_threshold=0.85,
            pinned={"rush_0@9.0": 0},  # 1 décimale au lieu de 3
        )


def test_pinning_and_banning_the_same_key_is_rejected():
    """Imposé et banni à la fois : ce n'est pas au système de trancher lequel
    des deux ordres du monteur l'emporte."""
    from src import nodes
    from src.models import AnalysisResult

    key = segment_key(SEGS[0].source_file, SEGS[0].start_time)
    analysis = AnalysisResult(
        segments=list(SEGS[:8]),
        total_rushes_duration=100.0,
        recommended_output_duration=60.0,
        summary="",
    ).model_dump(mode="json")

    with pytest.raises(ValueError, match="bannie"):
        nodes._candidates(
            analysis=analysis,
            presets=[PRESETS["punchy"].model_dump()],
            k_per_preset=1,
            time_limit_s=3.0,
            banned=[key],
            dedupe_threshold=0.85,
            pinned={key: 0},
        )


def test_segment_key_matches_what_the_report_prints():
    from src.assemble import explain

    preset = PRESETS["punchy"]
    c = solve(SEGS, preset, k=1, time_limit_s=5.0)[0]
    report = explain(c, SEGS, preset)
    for key in keys_of(c, SEGS):
        assert key in report, f"{key} absent du rapport, donc non copiable"


def test_manual_mode_makes_no_model_call():
    from src import nodes
    from src.models import AnalysisResult

    segs = list(SEGS)
    analysis = AnalysisResult(
        segments=segs, total_rushes_duration=300.0,
        recommended_output_duration=60.0, summary="",
    ).model_dump(mode="json")
    presets = [PRESETS["punchy"].model_dump(), PRESETS["contemplative"].model_dump()]
    candidates = nodes._candidates(
        analysis=analysis, presets=presets, k_per_preset=2,
        time_limit_s=5.0, banned=[], dedupe_threshold=0.85,
    )

    def explode(*a, **k):
        raise AssertionError("le mode manuel ne doit appeler aucun modèle")

    import src.agents.comparator as comparator_mod

    original = comparator_mod.ComparatorAgent
    comparator_mod.ComparatorAgent = explode
    try:
        out = nodes._ranked(
            analysis=analysis, candidates=candidates, presets=presets,
            model="x", max_candidates=6, per_preset=1, mode="manual",
        )
    finally:
        comparator_mod.ComparatorAgent = original

    assert out["mode"] == "manual"
    assert out["comparisons"] == 0
    # En manuel on ne pré-filtre pas : le monteur voit tout le faisceau.
    assert len(out["ranked"]) == min(len(candidates), 6) > 2
    assert len(out["plans"]) == len(out["ranked"])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
