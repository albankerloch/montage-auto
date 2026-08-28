"""COMPARATOR : « lequel des deux ? » au lieu de « note sur 1.0 ».

Pourquoi ce changement. Le CRITIC produisait un score absolu 0.0–1.0 comparé à
un seuil de 0.70. Deux problèmes : la note d'un LLM sur une échelle absolue
n'est pas stable d'un appel à l'autre (c'est exactement le reproche fait aux VLM
noteurs), et le seuil n'était justifié par rien. Une comparaison par paires
n'exige aucune calibration : elle ne demande qu'un ordre.

Il reçoit les frames de raccord (`src/video/cuts.py`), pas trois keyframes
prises au hasard dans le rendu : il juge enfin le montage et pas le contenu.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent
from src.assemble import Candidate, Preset, to_edit_plan
from src.config import COMPARATOR_MODEL, COMPARATOR_MAX_CUTS
from src.models import EditPlan, VideoSegment
from src.video.cuts import cut_vision_blocks

SYSTEM_PROMPT = """Tu es monteur. On te présente DEUX montages candidats construits à partir des mêmes rushes.

Pour chacun : le plan de montage (durées, ordre, transitions, rôle et émotion de chaque plan) et des images de raccord — pour chaque coupe montrée, la dernière image du plan sortant puis la première du plan entrant.

Choisis lequel des deux tu livrerais. Critères, par ordre d'importance :
1. Les raccords fonctionnent-ils ? (saut d'axe, faux raccord, valeurs de plan identiques collées)
2. Le rythme : la durée des plans varie-t-elle avec l'intention, ou est-ce mécanique ?
3. L'arc : l'ordre construit-il quelque chose, ou est-ce une suite de bons plans ?

Tu DOIS choisir. Pas d'égalité, pas de « ça dépend ». Si les deux se valent, tranche
sur les raccords. Justifie en une phrase, en pointant un raccord ou un enchaînement précis."""


class Comparison(BaseModel):
    winner: Literal["A", "B"]
    reason: str = Field(description="Une phrase, citant un raccord ou un enchaînement précis")
    margin: Literal["nette", "faible"]


def _plan_digest(plan: EditPlan) -> str:
    """Résumé compact : on n'envoie pas l'EditPlan complet (segments verbeux)."""
    rows = [
        {
            "n": e.order,
            "src": e.segment.source_file.rsplit("/", 1)[-1],
            "dur": round(e.segment.duration, 2),
            "role": e.segment.suggested_role,
            "emo": e.segment.emotion,
            "q": round(e.segment.quality_score, 2),
            "trans": e.transition_in,
        }
        for e in sorted(plan.edits, key=lambda e: e.order)
    ]
    return json.dumps(rows, ensure_ascii=False)


class ComparatorAgent(BaseAgent):
    def __init__(self, model: str = COMPARATOR_MODEL):
        super().__init__(model=model)

    def compare(
        self,
        a: Candidate,
        b: Candidate,
        segments: list[VideoSegment],
        presets: dict[str, Preset],
        max_cuts: int = COMPARATOR_MAX_CUTS,
    ) -> Comparison:
        plan_a = to_edit_plan(a, segments, presets[a.preset], title="A")
        plan_b = to_edit_plan(b, segments, presets[b.preset], title="B")

        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"## Montage A\n{len(plan_a.edits)} plans, "
                    f"{plan_a.total_duration:.1f}s\n```json\n{_plan_digest(plan_a)}\n```"
                ),
            }
        ]
        content += cut_vision_blocks(plan_a, tag="cmp_A", label="A", max_cuts=max_cuts)
        content.append(
            {
                "type": "text",
                "text": (
                    f"\n## Montage B\n{len(plan_b.edits)} plans, "
                    f"{plan_b.total_duration:.1f}s\n```json\n{_plan_digest(plan_b)}\n```"
                ),
            }
        )
        content += cut_vision_blocks(plan_b, tag="cmp_B", label="B", max_cuts=max_cuts)
        content.append(
            {"type": "text", "text": "\nLequel livres-tu ? Réponds via l'outil."}
        )

        return self.call(
            system=SYSTEM_PROMPT,
            user_content=content,
            output_schema=Comparison,
            max_tokens=512,
        )

    def as_callable(self, segments: list[VideoSegment], presets: dict[str, Preset]):
        """Adaptateur pour `src.beam.rank`."""

        def _cmp(a: Candidate, b: Candidate) -> tuple[str, str]:
            verdict = self.compare(a, b, segments, presets)
            label_a = f"{a.preset}#{a.rank_in_preset}"
            label_b = f"{b.preset}#{b.rank_in_preset}"
            winner = label_a if verdict.winner == "A" else label_b
            return verdict.winner, f"{winner} ({verdict.margin}) — {verdict.reason}"

        return _cmp
