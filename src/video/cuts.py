"""Frames de raccord : la dernière image avant une coupe, la première après.

Le CRITIC historique recevait 3 keyframes à 10/50/90 % du rendu — il ne voyait
donc jamais une seule coupe, et jugeait le *contenu* en croyant juger le
*montage*. Ici on échantillonne les raccords eux-mêmes, directement dans les
rushes source : pas besoin d'avoir rendu la vidéo pour évaluer un plan.
"""
from __future__ import annotations

from pathlib import Path

from src.models import EditPlan
from src.video.thumbnails import extract_thumbnail, thumbnail_to_base64

_EPS = 0.06  # recul/avance en secondes autour de la frontière


def cut_times(plan: EditPlan, max_cuts: int = 4) -> list[tuple[int, str, float, str, float]]:
    """(index_coupe, fichier_sortant, t_sortant, fichier_entrant, t_entrant)."""
    edits = sorted(plan.edits, key=lambda e: e.order)
    if len(edits) < 2:
        return []

    n_cuts = len(edits) - 1
    if n_cuts <= max_cuts:
        picks = list(range(n_cuts))
    else:
        step = n_cuts / max_cuts
        picks = sorted({int(i * step) for i in range(max_cuts)})

    out = []
    for c in picks:
        out_seg, in_seg = edits[c].segment, edits[c + 1].segment
        t_out = max(out_seg.start_time, out_seg.end_time - _EPS)
        t_in = min(in_seg.end_time, in_seg.start_time + _EPS)
        out.append((c, out_seg.source_file, t_out, in_seg.source_file, t_in))
    return out


def cut_frame_paths(plan: EditPlan, tag: str, max_cuts: int = 4) -> list[tuple[int, str, str]]:
    """Extrait les paires de frames de raccord. Renvoie (index, avant, après)."""
    pairs: list[tuple[int, str, str]] = []
    for c, f_out, t_out, f_in, t_in in cut_times(plan, max_cuts):
        try:
            before = extract_thumbnail(f_out, t_out, f"{tag}_cut{c:02d}_out", width=384)
            after = extract_thumbnail(f_in, t_in, f"{tag}_cut{c:02d}_in", width=384)
        except Exception as e:  # noqa: BLE001 — un raccord manquant n'invalide pas le plan
            print(f"[CUTS] raccord {c} non extrait ({Path(f_out).name}): {e}")
            continue
        pairs.append((c, before, after))
    return pairs


def cut_vision_blocks(plan: EditPlan, tag: str, label: str, max_cuts: int = 4) -> list[dict]:
    """Blocs `content` prêts pour l'API, montrant chaque raccord."""
    blocks: list[dict] = []
    for c, before, after in cut_frame_paths(plan, tag, max_cuts):
        edits = sorted(plan.edits, key=lambda e: e.order)
        blocks.append(
            {
                "type": "text",
                "text": (
                    f"\n**{label} — raccord {c + 1}** "
                    f"({edits[c].segment.duration:.1f}s → {edits[c + 1].segment.duration:.1f}s, "
                    f"transition « {edits[c + 1].transition_in} »). "
                    "Image sortante puis image entrante :"
                ),
            }
        )
        for path in (before, after):
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": thumbnail_to_base64(path),
                    },
                }
            )
    return blocks
