"""Compare deux modèles d'annotation sur les mêmes rushes.

Écrit parce que la question « est-ce qu'un VLM local suffit ? » ne se répond
pas par une opinion. Le graphe rend la comparaison presque gratuite : les deux
modèles partagent `scenes`, `thumbs` (à largeur égale) et `metrics`, seul le
nœud `annot` diffère. On matérialise les deux et on compare.

    python -m src.bench_annot rushes/ \\
        --a claude-haiku-4-5-20251001 \\
        --b local/Qwen/Qwen3-VL-8B-Instruct

Trois niveaux de mesure, du moins au plus significatif :

  1. **accord champ à champ** (rôle, émotion) et corrélation des notes
     d'intérêt. Facile à lire, mais un désaccord sur `emotion` n'a pas
     forcément de conséquence ;
  2. **accord sur la sélection** : les deux annotations produisent-elles le
     même montage ? C'est ce qui compte, et c'est plus indulgent — le solveur
     absorbe une partie du bruit ;
  3. **rang de corrélation des notes**, qui dit si le modèle local ordonne les
     plans comme le modèle de référence même s'il ne note pas sur la même
     échelle.

Le classement par comparaison est volontairement mis en mode manuel : faire
intervenir le comparateur ajouterait une source de variance sans rapport avec
la question posée.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Sequence

from src.assemble import Candidate
from src.models import AnalysisResult, segment_key


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Corrélation de rang, sans dépendance à scipy.

    De rang et non de Pearson : deux modèles n'utilisent pas la même échelle
    (l'un note serré autour de 0.6, l'autre étale sur [0.2, 0.9]) et ce qui
    importe au solveur est l'ordre, pas la valeur absolue.
    """
    n = len(xs)
    if n < 3:
        return float("nan")

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def _by_key(analysis: AnalysisResult) -> dict:
    return {segment_key(s.source_file, s.start_time): s for s in analysis.segments}


def _selection(candidates: list[dict], segments) -> set[str]:
    keys: set[str] = set()
    for raw in candidates:
        c = Candidate.model_validate(raw)
        for p in c.picks:
            s = segments[p.index]
            keys.add(segment_key(s.source_file, s.start_time))
    return keys


def compare(run_a, run_b, label_a: str, label_b: str) -> str:
    a = AnalysisResult.model_validate(run_a.get("segments"))
    b = AnalysisResult.model_validate(run_b.get("segments"))
    ka, kb = _by_key(a), _by_key(b)
    common = sorted(set(ka) & set(kb))

    lines = [
        f"A = {label_a}",
        f"B = {label_b}",
        "",
        f"{len(ka)} plans annotés par A, {len(kb)} par B, {len(common)} en commun",
    ]
    if len(ka) != len(kb):
        lines.append(
            "  ⚠ écart de volume : un des deux a échoué sur certains plans "
            "(les annotations en échec sont écartées)"
        )
    if not common:
        return "\n".join(lines + ["", "Aucun plan comparable."])

    role = sum(ka[k].suggested_role == kb[k].suggested_role for k in common) / len(common)
    emo = sum(ka[k].emotion == kb[k].emotion for k in common) / len(common)
    qa = [ka[k].quality_score for k in common]
    qb = [kb[k].quality_score for k in common]
    rho = _spearman(qa, qb)
    bias = statistics.fmean(qb) - statistics.fmean(qa)

    lines += [
        "",
        "── Accord champ à champ ────────────────────────────────",
        f"  rôle narratif identique   : {role:.0%}",
        f"  émotion identique         : {emo:.0%}",
        f"  intérêt, corrélation rang : {rho:+.2f}"
        + ("  (l'ordre est préservé)" if rho > 0.6 else "  (ordres divergents)"),
        f"  intérêt, biais moyen B−A  : {bias:+.2f}"
        + ("  (B note plus haut)" if bias > 0.05 else
           "  (B note plus bas)" if bias < -0.05 else "  (échelles comparables)"),
        f"  intérêt, écart-type A/B   : {statistics.pstdev(qa):.2f} / {statistics.pstdev(qb):.2f}"
        + ("  (B écrase la dynamique)" if statistics.pstdev(qb) < statistics.pstdev(qa) * 0.6 else ""),
    ]

    sel_a = _selection(run_a.get("candidates"), a.segments)
    sel_b = _selection(run_b.get("candidates"), b.segments)
    inter = len(sel_a & sel_b)
    union = len(sel_a | sel_b)
    lines += [
        "",
        "── Accord sur la sélection ─────────────────────────────",
        "  (ce qui compte : le solveur absorbe une partie du bruit d'annotation)",
        f"  plans retenus par A : {len(sel_a)}",
        f"  plans retenus par B : {len(sel_b)}",
        f"  Jaccard             : {inter / union:.0%}" if union else "  Jaccard : —",
    ]
    only_a = sorted(sel_a - sel_b)[:5]
    only_b = sorted(sel_b - sel_a)[:5]
    if only_a:
        lines.append(f"  retenus par A seul  : {', '.join(only_a)}")
    if only_b:
        lines.append(f"  retenus par B seul  : {', '.join(only_b)}")

    lines += [
        "",
        "── Lecture ─────────────────────────────────────────────",
    ]
    if union and inter / union > 0.7 and rho > 0.6:
        lines.append("  B produit une sélection très proche de A : substituable ici.")
    elif union and inter / union > 0.45:
        lines.append(
            "  B diverge sensiblement. Regarder si les plans « retenus par A seul »\n"
            "  sont meilleurs à l'œil avant de conclure — A n'est pas la vérité."
        )
    else:
        lines.append(
            "  B produit un montage différent. Ce n'est pas forcément pire :\n"
            "  seul un visionnage, ou le taux d'accord de src.conform sur un vrai\n"
            "  montage livré, peut trancher."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare deux modèles d'annotation sur les mêmes rushes",
    )
    parser.add_argument("rushes", nargs="+")
    parser.add_argument("--a", required=True, help="Modèle de référence")
    parser.add_argument("--b", required=True, help="Modèle à évaluer (ex. local/…)")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--presets", type=str, default="punchy,emotional_arc")
    parser.add_argument(
        "--width", type=int, default=640,
        help="Largeur des vignettes, identique pour les deux : comparer un "
             "modèle à 1280 px avec un autre à 640 px ne mesure rien",
    )
    args = parser.parse_args(argv)

    from src.main import collect_rush_paths
    from src.pipeline import Run

    paths = collect_rush_paths(args.rushes)
    if not paths:
        print("Aucun rush trouvé", file=sys.stderr)
        return 1

    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    runs = {}
    for label, model in (("A", args.a), ("B", args.b)):
        print(f"[{label}] {model} …", flush=True)
        t0 = time.time()
        r = Run(
            paths, preset_names=presets, target_duration=args.duration,
            annot_model=model, thumbnail_width=args.width,
            # Isole la variable : pas d'appel au comparateur, dont la variance
            # n'a rien à voir avec la question posée. Ni veto ni pin non plus —
            # une contrainte humaine forcerait les deux modèles à converger et
            # ferait paraître l'écart plus faible qu'il n'est.
            rank_mode="manual",
            verbose=False,
        )
        r.get("candidates")
        print(f"[{label}] {time.time() - t0:.1f}s "
              f"(cache compris — relancer pour mesurer le temps de calcul seul)")
        runs[label] = r

    print()
    print(compare(runs["A"], runs["B"], args.a, args.b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
