"""Bout en bout du moteur graphe, LLM simulés.

Couvre ce que les tests unitaires ne touchaient pas : détection de scènes,
vignettes, fusion en AnalysisResult, appel du solveur depuis un nœud, classement,
export EDL/FCPXML, rendu moviepy — et surtout la sémantique du cache
(qu'est-ce qui est réellement invalidé quand on change quoi).

Les deux agents LLM sont remplacés par des doubles déterministes : ces tests
tournent sans ANTHROPIC_API_KEY. Ils exigent en revanche ffmpeg et moviepy, et
sont donc marqués `slow`.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import beam, nodes  # noqa: E402
from src.agents.annotator import SegmentSemantics  # noqa: E402
from src.assemble import Candidate  # noqa: E402

RUSHES_DIR = Path(__file__).parent / "fixtures" / "rushes"
pytestmark = pytest.mark.skipif(
    not RUSHES_DIR.is_dir() or not list(RUSHES_DIR.glob("*.mp4")),
    reason="fixtures vidéo absentes — voir tests/fixtures/README.md",
)

ROLES = ["opening", "build_up", "climax", "resolution", "outro", "b_roll"]
EMOTIONS = ["energetic", "joyful", "tense", "neutral", "calm", "melancholic"]


def fake_annotate(rush: str, scenes, thumbs, model: str) -> list[dict]:
    """Double de l'ANNOTATOR : dérivé du hash, donc stable d'un run à l'autre.

    Un double non déterministe rendrait le cache menteur — c'est exactement la
    propriété qu'on veut vérifier ici.
    """
    out = []
    for i, (s, e) in enumerate(scenes):
        h = hashlib.sha256(f"{Path(rush).name}:{i}".encode()).digest()
        out.append(
            SegmentSemantics(
                segment_index=i,
                quality_score=round(0.2 + (h[0] / 255) * 0.75, 2),
                semantic_tags=["b_roll"],
                emotion=EMOTIONS[h[1] % len(EMOTIONS)],
                suggested_role=ROLES[h[2] % len(ROLES)],
                include_recommendation=True,
            ).model_dump(mode="json")
        )
    return out


def fake_rank(analysis, candidates, presets, model, max_candidates, per_preset):
    """Double du COMPARATOR : préfère le candidat au plus grand nombre de plans."""
    from src.assemble import Preset, to_edit_plan
    from src.models import AnalysisResult

    segs = AnalysisResult.model_validate(analysis).segments
    preset_map = {p["name"]: Preset.model_validate(p) for p in presets}
    cands = beam.objective_prefilter(
        [Candidate.model_validate(c) for c in candidates], per_preset=per_preset
    )
    ranked, calls = beam.rank(
        cands,
        lambda a, b: ("A" if len(a.picks) > len(b.picks) else "B", "double"),
        max_candidates=max_candidates,
    )
    return {
        "comparisons": calls,
        "ranked": [r.model_dump(mode="json") for r in ranked],
        "plans": [
            to_edit_plan(
                r.candidate, segs, preset_map[r.candidate.preset],
                title=f"{r.rank + 1:02d}_{r.candidate.preset}",
            ).model_dump(mode="json")
            for r in ranked
        ],
    }


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(nodes, "_annot", fake_annotate)
    monkeypatch.setattr(nodes, "_ranked", fake_rank)


@pytest.fixture
def make_run(tmp_path, stubbed):
    from src.pipeline import Run

    def _make(**kw):
        return Run(
            [str(p) for p in sorted(RUSHES_DIR.glob("*.mp4"))],
            preset_names=kw.pop("preset_names", ["punchy", "best_of"]),
            target_duration=kw.pop("target_duration", 20.0),
            max_segments_per_rush=kw.pop("max_segments_per_rush", 8),
            solver_time_limit_s=kw.pop("solver_time_limit_s", 5.0),
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "out",
            verbose=False,
            **kw,
        )

    return _make


def test_segments_are_built_from_real_video(make_run):
    from src.models import AnalysisResult

    r = make_run()
    analysis = AnalysisResult.model_validate(r.get("segments"))
    assert len(analysis.segments) >= 12
    for s in analysis.segments:
        assert s.end_time > s.start_time
        assert Path(s.thumbnail_path).exists(), "vignette non matérialisée"
        assert 0.0 <= s.quality_score <= 1.0


def test_plan_bounds_stay_inside_source_media(make_run):
    """Le clamp anti-hallucination du dépôt vaut aussi pour le solveur."""
    from src.models import AnalysisResult, EditPlan
    from src.video.probe import probe_video

    r = make_run()
    analysis = AnalysisResult.model_validate(r.get("segments"))
    durations = {s.source_file: probe_video(s.source_file).duration for s in analysis.segments}
    for raw in r.get("ranked")["plans"]:
        for e in EditPlan.model_validate(raw).edits:
            assert e.segment.start_time >= 0.0
            assert e.segment.end_time <= durations[e.segment.source_file] + 1e-6


def test_second_run_is_a_full_cache_hit(make_run):
    r1 = make_run()
    r1.get("ranked")
    r2 = make_run()
    assert r2.todo("ranked") == [], "un run identique doit tout retrouver en cache"


def test_changing_duration_spares_the_vision_nodes(make_run):
    """L'argument central du graphe : la vision est le poste de coût, et
    changer un paramètre de montage ne doit pas la refaire."""
    r1 = make_run(target_duration=20.0)
    r1.get("ranked")

    r2 = make_run(target_duration=24.0)
    names = {n.name for n in r2.todo("ranked")}
    assert names == {"candidates", "ranked"}
    assert "annot" not in names and "thumbs" not in names and "scenes" not in names


def test_adding_a_rush_spares_the_other_rushes(make_run, tmp_path):
    import shutil

    r1 = make_run()
    r1.get("segments")

    extra_dir = tmp_path / "more"
    extra_dir.mkdir()
    files = sorted(RUSHES_DIR.glob("*.mp4"))
    for f in files:
        shutil.copy2(f, extra_dir / f.name)
    shutil.copy2(files[0], extra_dir / "rush_extra.mp4")

    from src.pipeline import Run

    r2 = Run(
        [str(p) for p in sorted(extra_dir.glob("*.mp4"))],
        preset_names=["punchy", "best_of"],
        target_duration=20.0,
        max_segments_per_rush=8,
        solver_time_limit_s=5.0,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        verbose=False,
    )
    todo = [n for n in r2.todo("segments")]
    # Le nouveau fichier a un chemin distinct donc ses nœuds sont neufs ;
    # les rushes déjà vus ne sont PAS réanalysés (même empreinte de contenu).
    assert sum(1 for n in todo if n.name == "annot") <= 2, [str(n) for n in todo]


def test_exports_and_render_produce_real_files(make_run):
    r = make_run()
    exports = r.get("exports")
    # EDL + FCPXML + le plan JSON, qui est l'entrée de `python -m src.export_resolve`.
    assert len(exports) == 3
    assert {Path(p).suffix.lower() for p in exports} == {".edl", ".fcpxml", ".json"}
    for p in exports:
        assert Path(p).stat().st_size > 0

    edl = next(p for p in exports if p.endswith(".edl"))
    assert "CMX" in Path(edl).read_text() or "TITLE" in Path(edl).read_text()

    out = r.get("render")
    assert Path(out).exists() and Path(out).stat().st_size > 10_000


def test_render_duration_matches_the_plan(make_run):
    from src.models import EditPlan
    from src.video.probe import probe_video

    r = make_run()
    plan = EditPlan.model_validate(r.get("ranked")["plans"][0])
    actual = probe_video(r.get("render")).duration
    assert abs(actual - plan.total_duration) < 1.0, f"{actual}s rendu vs {plan.total_duration}s planifié"
