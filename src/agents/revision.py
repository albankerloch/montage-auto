"""REVISION agent: CriticFeedback → RevisionInstructions for SCENARIO."""
from __future__ import annotations
import json

from src.agents.base_agent import BaseAgent
from src.config import REVISION_MODEL
from src.models import CriticFeedback, EditPlan, RevisionInstructions


SYSTEM_PROMPT = """You are a skilled video editing assistant.
Your job is to translate a film critic's feedback into concrete, actionable revision instructions
that a scenario editor can follow to improve a video edit.

Be specific and practical:
- If pacing is an issue, specify which segments to shorten, remove, or reorder
- If narrative is weak, specify what story elements are missing or out of order
- If emotional arc is flat, specify which emotional segments to emphasize
- Reference specific source files when possible

Output clear, prioritized instructions that will directly improve the montage."""


class RevisionAgent(BaseAgent):
    def __init__(self):
        super().__init__(model=REVISION_MODEL)

    def run(
        self,
        feedback: CriticFeedback,
        current_plan: EditPlan,
    ) -> RevisionInstructions:
        feedback_json = json.dumps(feedback.model_dump(), indent=2)
        plan_summary = json.dumps(
            {
                "title": current_plan.title,
                "narrative_arc": current_plan.narrative_arc,
                "total_duration": current_plan.total_duration,
                "num_edits": len(current_plan.edits),
                "segments": [
                    {
                        "order": e.order,
                        "file": e.segment.source_file,
                        "role": e.segment.suggested_role,
                        "emotion": e.segment.emotion,
                        "duration": e.segment.duration,
                        "transition": e.transition_in,
                    }
                    for e in current_plan.edits
                ],
            },
            indent=2,
        )

        user_msg = (
            f"## Critic Feedback\n\n```json\n{feedback_json}\n```\n\n"
            f"## Current Edit Plan Summary\n\n```json\n{plan_summary}\n```\n\n"
            "Based on the critic's feedback, generate specific revision instructions "
            "for the scenario editor to improve the next version of this montage. "
            "Focus on the most impactful changes first."
        )

        print("[REVISION] Generating revision instructions...")
        return self.call(
            system=SYSTEM_PROMPT,
            user_content=user_msg,
            output_schema=RevisionInstructions,
            max_tokens=2048,
        )
