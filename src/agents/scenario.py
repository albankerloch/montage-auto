"""SCENARIO agent: segments → EditPlan (timeline)."""
from __future__ import annotations
import json

from src.agents.base_agent import BaseAgent
from src.config import SCENARIO_MODEL
from src.models import AnalysisResult, EditPlan, RevisionInstructions


SYSTEM_PROMPT = """You are an expert film editor with deep knowledge of narrative structure, pacing, and visual storytelling.

Your task is to create a compelling video edit plan (timeline) from analyzed video segments.

Core editing principles to apply:
- Three-act structure: setup → confrontation/build → resolution
- Vary shot duration for pacing (short cuts = energy, longer cuts = reflection)
- Prioritize high-quality segments (quality_score > 0.6)
- Use the segment's suggested_role to guide placement
- Create audio continuity across cuts
- The opening should hook the viewer immediately
- Match emotional arc: build from neutral → peak emotion → resolution

Transition guidelines:
- "cut": instant cut, use for energy and fast pacing
- "fade": fade to/from black, use for time jumps or emotional weight
- "dissolve": overlap transition, use for smooth flow between similar scenes

Return an EditPlan with the segments ordered for maximum narrative impact."""


class ScenarioAgent(BaseAgent):
    def __init__(self):
        super().__init__(model=SCENARIO_MODEL)

    def run(
        self,
        analysis: AnalysisResult,
        revision_instructions: RevisionInstructions | None = None,
    ) -> EditPlan:
        # Serialize segments for Claude
        segments_json = json.dumps(
            [s.model_dump(exclude={"thumbnail_path"}) for s in analysis.segments],
            indent=2,
        )

        user_msg = (
            f"## Available Video Segments\n\n```json\n{segments_json}\n```\n\n"
            f"## Context\n"
            f"- Total rushes duration: {analysis.total_rushes_duration:.1f}s\n"
            f"- Recommended output duration: {analysis.recommended_output_duration:.1f}s\n"
            f"- Summary: {analysis.summary}\n\n"
            "Create an EditPlan that tells a compelling story. "
            "You don't need to use all segments — select the best ones. "
            "Ensure the total_duration in your plan matches the sum of segment durations "
            "accounting for speed_factor adjustments.\n"
        )

        if revision_instructions:
            revisions_json = json.dumps(revision_instructions.model_dump(), indent=2)
            user_msg += (
                f"\n## REVISION INSTRUCTIONS (apply these changes)\n\n"
                f"```json\n{revisions_json}\n```\n\n"
                "Your previous edit plan was reviewed and needs improvement. "
                "Please apply the revision instructions above carefully."
            )

        print("[SCENARIO] Generating edit plan...")
        return self.call(
            system=SYSTEM_PROMPT,
            user_content=user_msg,
            output_schema=EditPlan,
            max_tokens=4096,
        )
