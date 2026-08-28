"""Point d'entrée du moteur « graphe + solveur + faisceau ».

Comparé à `orchestrator.py` (machine à états avec boucle de révision) :

  orchestrator : START → ANALYZE → SCENARIO → EDIT → CRITIC ⟲ REVISION → QUALITY
                 un ordre écrit à la main, un artefact unique en sortie,
                 rien de réutilisable entre deux runs.

  pipeline     : on demande un artefact, le graphe remonte ses dépendances.
                 L'assemblage est un problème d'optimisation résolu en un
                 passage, le faisceau explore en parallèle, la sortie est un
                 classement — pas un verdict.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Sequence

from src import nodes
from src.assemble import Candidate, Preset, explain
from src.config import (
    ANALYZER_MODEL,
    CACHE_DIR,
    COMPARATOR_MODEL,
    DEDUPE_THRESHOLD,
    K_PER_PRESET,
    MAX_CANDIDATES,
    MAX_SEGMENTS_PER_RUSH,
    OUTPUT_DIR,
    SOLVER_TIME_LIMIT_S,
    TARGET_MONTAGE_DURATION,
)
from src.graph import Node, Store, materialize, render_plan, stale
from src.models import AnalysisResult


class Run:
    """Un run = un graphe + un cache. Pas d'état mutable qui circule."""

    def __init__(
        self,
        rush_paths: Sequence[str],
        *,
        preset_names: Sequence[str],
        target_duration: float = TARGET_MONTAGE_DURATION,
        max_segments_per_rush: int = MAX_SEGMENTS_PER_RUSH,
        k_per_preset: int = K_PER_PRESET,
        solver_time_limit_s: float = SOLVER_TIME_LIMIT_S,
        max_candidates: int = MAX_CANDIDATES,
        banned_segments: Sequence[str] = (),
        cache_dir: str | Path = CACHE_DIR,
        output_dir: str | Path = OUTPUT_DIR,
        annot_model: str = ANALYZER_MODEL,
        comparator_model: str = COMPARATOR_MODEL,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.output_dir = Path(output_dir)
        self.store = Store(cache_dir)
        self.presets: list[Preset] = nodes.resolve_presets(
            preset_names, target_duration=target_duration
        )
        self.nodes: dict[str, Node] = nodes.build(
            rush_paths,
            presets=self.presets,
            annot_model=annot_model,
            comparator_model=comparator_model,
            output_dir=str(output_dir),
            max_segments_per_rush=max_segments_per_rush,
            k_per_preset=k_per_preset,
            solver_time_limit_s=solver_time_limit_s,
            banned_segments=banned_segments,
            dedupe_threshold=DEDUPE_THRESHOLD,
            max_candidates=max_candidates,
            target_duration=target_duration,
        )

    # ── introspection ───────────────────────────────────────────────────────

    def plan(self, target: str = "ranked") -> str:
        return render_plan(self.nodes[target], self.store)

    def todo(self, target: str = "ranked") -> list[Node]:
        """Ce qui serait recalculé — le reste est déjà en cache."""
        return stale(self.nodes[target], self.store)

    # ── exécution ───────────────────────────────────────────────────────────

    def get(self, target: str):
        node = self.nodes[target]
        t0 = time.time()
        hits = {"hit": 0, "done": 0}

        def on_event(kind: str, n: Node, detail: str = "") -> None:
            if kind in hits:
                hits[kind] += 1
            if not self.verbose:
                return
            if kind == "miss":
                # On ne journalise que les calculs : lister les dizaines de
                # nœuds servis par le cache noie le signal à chaque appel.
                print(f"  ⟳ {n.name}:{n.key()[:8]}" + (f" [{n.label}]" if n.label else ""))

        value = materialize(node, self.store, on_event)
        if self.verbose:
            print(
                f"[{target}] {hits['done']} calculé(s), {hits['hit']} depuis le cache, "
                f"{time.time() - t0:.1f}s"
            )
        return value

    def deliver(self, targets: Sequence[str] = ("render", "exports")) -> list[str]:
        """Recopie les livrables du cache vers `output/`.

        Le cache est indexé par clé, donc illisible pour un humain : les
        artefacts y vivent sous `render/<hash>/…`. Ce qu'on livre au monteur
        doit atterrir à un endroit stable et nommable.
        """
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        delivered: list[str] = []
        for t in targets:
            value = self.get(t)
            for src in [value] if isinstance(value, str) else value:
                dest = out_dir / Path(src).name
                if Path(src).resolve() != dest.resolve():
                    shutil.copy2(src, dest)
                delivered.append(str(dest))
        return delivered

    # ── rapports ────────────────────────────────────────────────────────────

    def report_candidates(self) -> str:
        analysis = AnalysisResult.model_validate(self.get("segments"))
        raw = self.get("candidates")
        by_name = {p.name: p for p in self.presets}
        lines = [f"{len(raw)} candidat(s) après déduplication\n"]
        for c in (Candidate.model_validate(x) for x in raw):
            lines.append(explain(c, analysis.segments, by_name[c.preset]))
            lines.append("")
        return "\n".join(lines)

    def report_ranking(self) -> str:
        r = self.get("ranked")
        lines = [
            f"Classement — {r['comparisons']} comparaison(s) par paires "
            f"(aucune note absolue, aucun seuil)\n"
        ]
        for entry, plan in zip(r["ranked"], r["plans"]):
            c = entry["candidate"]
            tag = "→ LIVRÉ" if entry["rank"] == 0 else f"  alternate {entry['rank']}"
            lines.append(
                f"{tag}  {c['preset']}#{c['rank_in_preset']}  "
                f"{len(plan['edits'])} plans · {plan['total_duration']:.1f}s · "
                f"{entry['wins']} victoire(s)"
            )
            for n in entry["notes"][:1]:
                lines.append(f"          {n}")
        return "\n".join(lines)


def run(
    rush_paths: Sequence[str],
    preset_names: Sequence[str],
    *,
    render: bool = True,
    export: bool = True,
    **kwargs,
) -> Run:
    r = Run(rush_paths, preset_names=preset_names, **kwargs)

    print("\n── Graphe ──────────────────────────────────────────────")
    print(r.plan("ranked"))
    print(f"\n{len(r.todo('ranked'))} nœud(s) à calculer\n")

    print("── Annotation ──────────────────────────────────────────")
    r.get("segments")

    print("\n── Assemblage (CP-SAT) ─────────────────────────────────")
    print(r.report_candidates())

    print("── Classement (comparaison par paires) ─────────────────")
    print(r.report_ranking())

    targets = tuple(t for t, on in (("exports", export), ("render", render)) if on)
    if targets:
        print("\n── Livrables ───────────────────────────────────────────")
        for p in r.deliver(targets):
            print(f"  {p}")

    return r
