"""Définition du graphe d'artefacts.

C'est ici que le workflow cesse d'être une chaîne : on ne décrit aucun ordre
d'exécution, seulement qui dépend de quoi. Les nœuds par rush (`probe`,
`scenes`, `thumbs`, `annot`) sont indépendants entre rushes — ajouter un fichier
n'invalide que ses propres nœuds ; changer la durée cible n'invalide que
`candidates` et l'aval, jamais les annotations vision, qui sont le poste de coût.

Le monteur est une entrée du graphe comme une autre : `overrides` (segments
bannis, segments imposés) est un paramètre de `candidates`. Le corriger
invalide le plan, pas l'analyse — ce que la boucle ne savait pas faire.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from src.assemble import PRESETS, Candidate, Preset, to_edit_plan
from src.graph import Node, source
from src.models import AnalysisResult, EditPlan, MontageResult, VideoSegment, segment_key

# Versions de calcul. À incrémenter quand la *sémantique* d'un nœud change —
# c'est la migration du cache. Le corps des fonctions n'est pas haché.
V = {
    "probe": "1",
    "scenes": "1",
    "thumbs": "1",
    "annot": "3",  # 3 = le modèle ne juge plus netteté ni exposition
    "metrics": "2",  # 2 = mouvement caméra par RANSAC, stabilité optionnelle
    "segments": "3",  # 3 = fusion des métriques locales
    "candidates": "3",  # 3 = terme technique dans l'objectif  # 2 = exclusion par clé, plus par filtrage de liste
    "ranked": "3",  # 3 = mode manuel + source_key dans le plan
    "alternates": "1",
    "render": "1",
    "exports": "1",
}


# ── Fonctions de calcul ─────────────────────────────────────────────────────


def _probe(rush: str) -> dict:
    from src.video.probe import probe_video

    return probe_video(rush).model_dump(mode="json")


def _scenes(rush: str, threshold: float, max_segments: int, min_duration: float) -> list[list[float]]:
    from src.video.probe import detect_scenes

    scenes = [s for s in detect_scenes(rush, threshold=threshold) if s[1] - s[0] >= min_duration]
    if len(scenes) > max_segments:
        # Échantillonnage uniforme sur toute la durée. L'ancien `scenes[:10]`
        # gardait les dix PREMIÈRES scènes : sur un mariage de 40 min, le
        # système ne voyait jamais la fin.
        step = len(scenes) / max_segments
        scenes = [scenes[int(i * step)] for i in range(max_segments)]
    return [[float(s), float(e)] for s, e in scenes]


def _thumbs(rush: str, scenes: list[list[float]], offset: float) -> list[str]:
    from src.video.thumbnails import extract_thumbnail

    stem = Path(rush).stem
    out: list[str] = []
    for i, (s, e) in enumerate(scenes):
        t = s + (e - s) * offset
        out.append(extract_thumbnail(rush, t, f"{stem}_sc{i:03d}", width=640))
    return out


def _annot(rush: str, scenes: list[list[float]], thumbs: list[str], model: str) -> list[dict]:
    from src.agents.annotator import AnnotatorAgent

    agent = AnnotatorAgent(model=model)
    sem = agent.annotate(rush, [(s, e) for s, e in scenes], thumbs)
    return [x.model_dump(mode="json") for x in sem]


def _metrics(rush: str, scenes: list[list[float]], n_samples: int) -> list[dict]:
    """Mesures locales, en pleine résolution. Aucun réseau, aucun modèle.

    Nœud frère de `annot` et non successeur : les deux ne dépendent que de
    `scenes`, donc ils sont indépendants et parallélisables.
    """
    from src.video.metrics import measure_scenes

    out = measure_scenes(rush, [(s, e) for s, e in scenes], n_samples=n_samples)
    return [m.model_dump(mode="json") for m in out]


def _segments(n_rushes: int, drop_failed: bool, **arts: Any) -> dict:
    """Fusionne les nœuds par rush en un `AnalysisResult`.

    Ce nœud ne prend AUCUN paramètre de montage. C'est délibéré : il porte le
    résultat de la vision, qui est le poste de coût. Y glisser la durée cible
    (comme je l'avais fait d'abord) suffisait à faire réannoter tous les rushes
    dès qu'on passait de 60 à 90 s — soit exactement ce que le graphe est censé
    empêcher. Le test `test_changing_duration_spares_the_vision_nodes` existe
    pour attraper cette régression.
    """
    segments: list[VideoSegment] = []
    total = 0.0
    dropped = 0

    for i in range(n_rushes):
        rush = arts[f"rush_{i}"]
        scenes = arts[f"scenes_{i}"]
        thumbs = arts[f"thumbs_{i}"]
        annots = arts[f"annot_{i}"]
        total += float(arts[f"probe_{i}"]["duration"])

        mets = arts.get(f"metrics_{i}") or [None] * len(scenes)

        for j, ((s, e), thumb, a) in enumerate(zip(scenes, thumbs, annots)):
            if drop_failed and a.get("failed"):
                dropped += 1
                continue
            m = mets[j] if j < len(mets) else None
            if m is not None and m.get("failed"):
                m = None  # mesure ratée : on n'invente pas une note technique
            if m:
                known = [
                    m[k] for k in ("sharpness", "exposure", "stability")
                    if m.get(k) is not None
                ]
                tech = min(known) if known else None
            else:
                tech = None
            segments.append(
                VideoSegment(
                    source_file=rush,
                    start_time=float(s),
                    end_time=float(e),
                    duration=float(e - s),
                    quality_score=float(a["quality_score"]),
                    semantic_tags=list(a["semantic_tags"]),
                    emotion=str(a["emotion"]),
                    suggested_role=str(a["suggested_role"]),
                    thumbnail_path=thumb,
                    technical_score=tech,
                    sharpness=m["sharpness"] if m else None,
                    exposure=m["exposure"] if m else None,
                    stability=m.get("stability") if m else None,
                    motion=m.get("motion") if m else None,
                )
            )

    summary = f"{len(segments)} segments retenus sur {n_rushes} rush(es)"
    if dropped:
        summary += f" — {dropped} écarté(s) pour annotation en échec"

    return AnalysisResult(
        segments=segments,
        total_rushes_duration=total,
        recommended_output_duration=total,  # la cible appartient aux presets
        summary=summary,
    ).model_dump(mode="json")


def _candidates(
    analysis: dict,
    presets: list[dict],
    k_per_preset: int,
    time_limit_s: float,
    banned: list[str],
    dedupe_threshold: float,
) -> list[dict]:
    from src import beam
    from src.assemble import solve

    segs = AnalysisResult.model_validate(analysis).segments

    blocked: frozenset[str] = frozenset(banned)
    if blocked:
        available = {segment_key(s.source_file, s.start_time) for s in segs}
        unknown = sorted(blocked - available)
        if unknown:
            # Une clé qui ne matche rien est presque toujours une faute de
            # frappe. L'ignorer donnerait au monteur un montage identique et la
            # conviction d'avoir banni un plan : le pire des deux mondes.
            raise ValueError(
                "Clé(s) de veto inconnue(s) : " + ", ".join(unknown)
                + f"\n{len(available)} segments disponibles, p. ex. : "
                + ", ".join(sorted(available)[:3])
            )

    out: list[Candidate] = []
    for pdict in presets:
        # Les presets sont indépendants : aucune dépendance entre eux, donc
        # parallélisables tels quels (ThreadPool ou nœuds de graphe séparés).
        out.extend(
            solve(
                segs,
                Preset.model_validate(pdict),
                k=k_per_preset,
                time_limit_s=time_limit_s,
                excluded=blocked,
            )
        )
    return [c.model_dump(mode="json") for c in beam.dedupe(out, threshold=dedupe_threshold)]


def _ranked(
    analysis: dict,
    candidates: list[dict],
    presets: list[dict],
    model: str,
    max_candidates: int,
    per_preset: int,
    mode: str = "llm",
) -> dict:
    from src import beam
    from src.agents.comparator import ComparatorAgent

    segs = AnalysisResult.model_validate(analysis).segments
    preset_map = {p["name"]: Preset.model_validate(p) for p in presets}
    cands = [Candidate.model_validate(c) for c in candidates]
    if mode != "manual":
        # Le pré-filtre existe pour économiser des appels au comparateur. En
        # manuel il n'y en a aucun : on montre au monteur tout le faisceau.
        cands = beam.objective_prefilter(cands, per_preset=per_preset)

    if mode == "manual":
        # « Lequel des deux tu livrerais » est une question de monteur. En mode
        # manuel on ne la pose pas à un modèle : on sort les K candidats dans
        # l'ordre des presets demandés et le monteur tranche dans son NLE.
        ranked = [
            beam.RankedCandidate(candidate=c, rank=i, wins=0, notes=["classement manuel"])
            for i, c in enumerate(cands[:max_candidates])
        ]
        calls = 0
    elif len(cands) <= 1:
        ranked = [
            beam.RankedCandidate(candidate=c, rank=i, wins=0) for i, c in enumerate(cands)
        ]
        calls = 0
    else:
        agent = ComparatorAgent(model=model)
        ranked, calls = beam.rank(
            cands, agent.as_callable(segs, preset_map), max_candidates=max_candidates
        )

    return {
        "mode": mode,
        "comparisons": calls,
        "ranked": [r.model_dump(mode="json") for r in ranked],
        "plans": [
            to_edit_plan(
                Candidate.model_validate(r.candidate.model_dump()),
                segs,
                preset_map[r.candidate.preset],
                title=f"{r.rank + 1:02d}_{r.candidate.preset}",
            ).model_dump(mode="json")
            for r in ranked
        ],
    }


def _alternates(ranked: dict, output_dir: str) -> list[str]:
    """Exporte TOUS les candidats en EDL + FCPXML.

    C'est le support de l'arbitrage humain : le monteur importe les K
    timelines dans Resolve, les compare sur du mouvement et pas sur des images
    fixes, puis relance avec `--pick` et `--ban`.
    """
    from src.export import export_all

    out: list[str] = []
    for i, raw in enumerate(ranked["plans"]):
        plan = EditPlan.model_validate(raw)
        exports = export_all(plan, output_dir)
        out.extend(exports[k] for k in ("edl", "fcpxml"))
    return sorted(out)


def _render(ranked: dict, which: int, output_dir: str) -> str:
    from src.video.editor import execute_edit_plan

    plan = EditPlan.model_validate(ranked["plans"][which])
    out = str(Path(output_dir) / f"montage_{which:02d}_{plan.title}.mp4")
    result: MontageResult = execute_edit_plan(plan, out)
    return result.output_path


def _exports(ranked: dict, which: int, output_dir: str) -> list[str]:
    from src.export import export_all

    plan = EditPlan.model_validate(ranked["plans"][which])
    return sorted(export_all(plan, output_dir).values())


# ── Construction du graphe ──────────────────────────────────────────────────


def resolve_presets(names: Sequence[str], **overrides: Any) -> list[Preset]:
    """Applique les surcharges globales (durée cible…) aux presets choisis."""
    out = []
    for n in names:
        base = PRESETS[n].model_dump()
        base.update({k: v for k, v in overrides.items() if v is not None})
        base["name"] = n
        out.append(Preset.model_validate(base))
    return out


def build(
    rush_paths: Sequence[str],
    *,
    presets: Sequence[Preset],
    annot_model: str,
    comparator_model: str,
    output_dir: str,
    scene_threshold: float = 0.25,
    max_segments_per_rush: int = 40,
    min_scene_duration: float = 0.6,
    thumbnail_offset: float = 0.3,
    metric_samples: int = 3,
    k_per_preset: int = 2,
    solver_time_limit_s: float = 15.0,
    banned_segments: Sequence[str] = (),
    dedupe_threshold: float = 0.85,
    max_candidates: int = 6,
    per_preset_prefilter: int = 1,
    rank_mode: str = "llm",
    pick: int = 0,
    drop_failed_annotations: bool = True,
    target_duration: float = 60.0,
) -> dict[str, Node]:
    """Renvoie les nœuds terminaux nommés : `segments`, `candidates`, `ranked`,
    `render`, `exports`. On matérialise celui dont on a besoin, pas plus."""
    merge_deps: dict[str, Node] = {}

    for i, path in enumerate(sorted(rush_paths)):
        src = source(path, name="rush")
        probe = Node("probe", _probe, {"rush": src}, version=V["probe"], label=Path(path).name)
        scenes = Node(
            "scenes",
            _scenes,
            {"rush": src},
            params={
                "threshold": scene_threshold,
                "max_segments": max_segments_per_rush,
                "min_duration": min_scene_duration,
            },
            version=V["scenes"],
            label=Path(path).name,
        )
        thumbs = Node(
            "thumbs",
            _thumbs,
            {"rush": src, "scenes": scenes},
            params={"offset": thumbnail_offset},
            version=V["thumbs"],
            codec="path_list",
            label=Path(path).name,
        )
        metrics = Node(
            "metrics",
            _metrics,
            {"rush": src, "scenes": scenes},
            params={"n_samples": metric_samples},
            version=V["metrics"],
            label=Path(path).name,
        )
        annot = Node(
            "annot",
            _annot,
            {"rush": src, "scenes": scenes, "thumbs": thumbs},
            params={"model": annot_model},
            version=V["annot"],
            label=Path(path).name,
        )
        merge_deps |= {
            f"rush_{i}": src,
            f"probe_{i}": probe,
            f"scenes_{i}": scenes,
            f"thumbs_{i}": thumbs,
            f"annot_{i}": annot,
            f"metrics_{i}": metrics,
        }

    segments = Node(
        "segments",
        _segments,
        merge_deps,
        params={
            "n_rushes": len(rush_paths),
            "drop_failed": drop_failed_annotations,
        },
        version=V["segments"],
    )

    preset_dicts = [p.model_dump() for p in presets]

    candidates = Node(
        "candidates",
        _candidates,
        {"analysis": segments},
        params={
            "presets": preset_dicts,
            "k_per_preset": k_per_preset,
            "time_limit_s": solver_time_limit_s,
            "banned": sorted(banned_segments),
            "dedupe_threshold": dedupe_threshold,
        },
        version=V["candidates"],
    )

    ranked = Node(
        "ranked",
        _ranked,
        {"analysis": segments, "candidates": candidates},
        params={
            "presets": preset_dicts,
            "model": comparator_model,
            "max_candidates": max_candidates,
            "per_preset": per_preset_prefilter,
            "mode": rank_mode,
        },
        version=V["ranked"],
    )

    alternates = Node(
        "alternates",
        _alternates,
        {"ranked": ranked},
        params={"output_dir": output_dir},
        version=V["alternates"],
        codec="path_list",
    )

    render = Node(
        "render",
        _render,
        {"ranked": ranked},
        params={"which": pick, "output_dir": output_dir},
        version=V["render"],
        codec="path",
    )

    exports = Node(
        "exports",
        _exports,
        {"ranked": ranked},
        params={"which": pick, "output_dir": output_dir},
        version=V["exports"],
        codec="path_list",
    )

    return {
        "segments": segments,
        "candidates": candidates,
        "ranked": ranked,
        "alternates": alternates,
        "render": render,
        "exports": exports,
    }
