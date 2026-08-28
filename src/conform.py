"""Retour depuis le NLE : relire ce que le monteur a réellement gardé.

C'est le maillon qui manquait. Jusqu'ici l'information circulait dans un seul
sens — le système proposait, le monteur travaillait dans Resolve, et rien ne
revenait. Le seul jugement disponible était celui d'un modèle sur ses propres
propositions, c'est-à-dire un proxy s'auto-évaluant.

Ici on relit la timeline conformée et on la compare au plan proposé. Ce que ça
produit :

  - **immédiatement** : les plans écartés deviennent des clés de veto, les plans
    ajoutés des candidats à imposer, sans que le monteur ait rien à recopier ;
  - **à terme** : un taux d'accord mesuré. C'est le premier signal d'évaluation
    du dépôt qui ne soit pas l'avis d'un modèle sur son propre travail, et le
    seul moyen que `quality_weight`, `technical_weight` et les tables d'énergie
    cessent d'être des nombres choisis au jugé.

Usage :

    python -m src.conform output/01_punchy_plan.json monté.fcpxml
    python -m src.conform output/01_punchy_plan.json monté.fcpxml --write-bans bans.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlparse

from pydantic import BaseModel

from src.models import EditPlan, segment_key

_RATIONAL = re.compile(r"^(-?\d+)(?:/(\d+))?s$")
_OVERLAP_MIN = 0.5      # recouvrement minimal pour considérer que c'est le même plan
_TRIM_TOLERANCE = 0.30  # écart de durée au-delà duquel on parle de recadrage


def parse_time(value: str | None) -> float:
    """FCPXML exprime le temps en rationnels : '150/25s', '6s', '0s'."""
    if not value:
        return 0.0
    m = _RATIONAL.match(value.strip())
    if not m:
        return 0.0
    num = float(m.group(1))
    den = float(m.group(2)) if m.group(2) else 1.0
    return num / den if den else 0.0


class ConformedClip(BaseModel):
    source_file: str
    start: float          # entrée dans le média source
    end: float
    record_offset: float  # position sur la timeline
    order: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_fcpxml(path: str | Path) -> list[ConformedClip]:
    """Relit une timeline FCPXML, la nôtre ou celle réexportée par Resolve.

    Resolve n'écrit pas exactement les mêmes balises que notre exporteur
    (`asset-clip` plutôt que `clip`, parfois `ref-clip`), d'où la collecte par
    présence d'un attribut `ref` plutôt que par nom de balise.
    """
    root = ET.parse(str(path)).getroot()

    assets: dict[str, str] = {}
    for asset in root.iter("asset"):
        aid = asset.get("id")
        src = asset.get("src")
        if src is None:
            rep = asset.find("media-rep")
            src = rep.get("src") if rep is not None else None
        if aid and src:
            assets[aid] = unquote(urlparse(src).path) if src.startswith("file:") else src

    clips: list[ConformedClip] = []
    spine = next(root.iter("spine"), None)
    if spine is None:
        return clips

    for el in spine.iter():
        ref = el.get("ref")
        if ref is None or ref not in assets:
            continue
        if el.get("offset") is None and el.get("start") is None:
            continue
        start = parse_time(el.get("start"))
        dur = parse_time(el.get("duration"))
        clips.append(
            ConformedClip(
                source_file=assets[ref],
                start=start,
                end=start + dur,
                record_offset=parse_time(el.get("offset")),
                order=len(clips),
            )
        )

    clips.sort(key=lambda c: c.record_offset)
    for i, c in enumerate(clips):
        c.order = i
    return clips


class ConformReport(BaseModel):
    kept: list[str] = []
    dropped: list[str] = []
    trimmed: list[str] = []
    reordered: list[str] = []
    added: list[str] = []
    proposed: int = 0
    conformed: int = 0

    @property
    def agreement(self) -> float:
        """Part des plans proposés que le monteur a conservés."""
        return len(self.kept) / self.proposed if self.proposed else 0.0

    def render(self) -> str:
        lines = [
            f"{self.proposed} plans proposés, {self.conformed} dans la timeline montée",
            f"Accord : {len(self.kept)}/{self.proposed} plans conservés "
            f"({self.agreement:.0%})",
        ]
        for label, items in (
            ("écartés", self.dropped),
            ("recadrés", self.trimmed),
            ("déplacés", self.reordered),
            ("ajoutés par le monteur", self.added),
        ):
            if items:
                lines.append(f"\n{len(items)} {label} :")
                lines.extend(f"  {k}" for k in items)
        if not self.dropped and not self.added:
            lines.append("\nAucun veto à déduire : le monteur n'a rien retiré ni ajouté.")
        return "\n".join(lines)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Recouvrement temporel, rapporté à la plus courte des deux plages."""
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    shortest = max(1e-6, min(a1 - a0, b1 - b0))
    return inter / shortest


def diff(plan: EditPlan, conformed: list[ConformedClip]) -> ConformReport:
    """Compare le plan proposé à la timeline effectivement montée.

    L'appariement se fait par recouvrement et non par égalité : un monteur
    rogne presque toujours les bornes. Un plan raccourci de 20 % reste le même
    plan, et le compter comme « écarté puis ajouté » donnerait un taux d'accord
    faux.
    """
    report = ConformReport(proposed=len(plan.edits), conformed=len(conformed))
    edits = sorted(plan.edits, key=lambda e: e.order)
    used: set[int] = set()
    matches: list[tuple[int, int]] = []  # (rang plan, rang timeline)

    for e in edits:
        seg = e.segment
        key = seg.source_key or segment_key(seg.source_file, seg.start_time)
        best, best_score = None, 0.0
        for j, c in enumerate(conformed):
            if j in used or Path(c.source_file).name != Path(seg.source_file).name:
                continue
            score = _overlap(seg.start_time, seg.end_time, c.start, c.end)
            if score > best_score:
                best, best_score = j, score

        if best is None or best_score < _OVERLAP_MIN:
            report.dropped.append(key)
            continue

        used.add(best)
        matches.append((e.order, best))
        report.kept.append(key)

        c = conformed[best]
        if abs(c.duration - seg.duration) / max(1e-6, seg.duration) > _TRIM_TOLERANCE:
            report.trimmed.append(
                f"{key} {seg.duration:.1f}s → {c.duration:.1f}s"
            )

    # Un plan est « déplacé » si son rang relatif a changé : on compare l'ordre
    # des positions timeline à l'ordre du plan pour les plans conservés.
    timeline_ranks = [j for _, j in matches]
    if timeline_ranks != sorted(timeline_ranks):
        for (plan_order, j), expected in zip(matches, sorted(timeline_ranks)):
            if j != expected:
                report.reordered.append(report.kept[[m[0] for m in matches].index(plan_order)])

    for j, c in enumerate(conformed):
        if j not in used:
            report.added.append(segment_key(c.source_file, c.start))

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare une timeline montée au plan proposé, et en déduit les vetos",
    )
    parser.add_argument("plan", help="Le _plan.json exporté par le pipeline")
    parser.add_argument("timeline", help="Le FCPXML réexporté depuis le NLE")
    parser.add_argument(
        "--write-bans",
        metavar="FICHIER",
        help="Écrire les plans écartés dans un JSON réutilisable avec --ban-file",
    )
    args = parser.parse_args(argv)

    plan = EditPlan.model_validate_json(Path(args.plan).read_text(encoding="utf-8"))
    conformed = parse_fcpxml(args.timeline)
    if not conformed:
        print(f"Aucun plan lisible dans {args.timeline}", file=sys.stderr)
        return 1

    report = diff(plan, conformed)
    print(report.render())

    if args.write_bans:
        Path(args.write_bans).write_text(
            json.dumps(report.dropped, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"\n{len(report.dropped)} veto(s) écrit(s) dans {args.write_bans}\n"
            f"  python -m src.main rushes/ --ban-file {args.write_bans}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
