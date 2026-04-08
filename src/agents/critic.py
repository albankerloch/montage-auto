"""CRITIC agent: evaluates the montage and produces a quality score + feedback."""
from __future__ import annotations
import json

from src.agents.base_agent import BaseAgent
from src.config import CRITIC_MODEL, QUALITY_THRESHOLD
from src.models import CriticFeedback, EditPlan, MontageResult
from src.video.thumbnails import extract_thumbnail, thumbnail_to_base64


SYSTEM_PROMPT = """You are a seasoned film critic and professional video editor evaluating a video montage.

Evaluate the edit plan on these criteria:

1. **Narrative coherence** (0-30 pts): Does the edit tell a clear, engaging story? Is there a proper arc?
2. **Pacing** (0-25 pts): Are shot durations varied appropriately? Does the rhythm match the content?
3. **Emotional impact** (0-25 pts): Does the emotional progression make sense? Is there a satisfying peak and resolution?
4. **Technical quality** (0-20 pts): Are high-quality segments prioritized? Are transitions appropriate?

Sum to 100, then divide by 100 to get a score from 0.0 to 1.0.

Be honest and critical. A score of 0.7+ means the edit is ready for output.
Provide specific, actionable improvements if the score is below 0.7."""


class CriticAgent(BaseAgent):
    def __init__(self):
        super().__init__(model=CRITIC_MODEL)

    def run(self, montage: MontageResult, edit_plan: EditPlan) -> CriticFeedback:
        # Build analysis context
        plan_json = json.dumps(edit_plan.model_dump(exclude_none=True), indent=2)

        # Extract a few keyframes from the output video for visual evaluation
        vision_content = self._build_vision_content(montage, edit_plan)

        user_msg_parts = vision_content + [
            {
                "type": "text",
                "text": (
                    f"\n## Edit Plan (JSON)\n\n```json\n{plan_json}\n```\n\n"
                    f"## Rendered Video Stats\n"
                    f"- Output file: {montage.output_path}\n"
                    f"- Actual duration: {montage.actual_duration:.1f}s\n"
                    f"- Number of cuts: {len(edit_plan.edits)}\n"
                    f"- Narrative arc described: {edit_plan.narrative_arc}\n\n"
                    f"The quality threshold for acceptance is {QUALITY_THRESHOLD}. "
                    "Evaluate this montage and provide detailed feedback."
                ),
            }
        ]

        print("[CRITIC] Evaluating montage...")
        feedback: CriticFeedback = self.call(
            system=SYSTEM_PROMPT,
            user_content=user_msg_parts,
            output_schema=CriticFeedback,
            max_tokens=2048,
        )

        # Ensure passed flag is consistent with score
        feedback.passed = feedback.score >= QUALITY_THRESHOLD
        print(f"[CRITIC] Score: {feedback.score:.2f} ({'PASSED' if feedback.passed else 'FAILED'})")
        return feedback

    def _build_vision_content(self, montage: MontageResult, edit_plan: EditPlan) -> list[dict]:
        """Extract keyframes from the rendered video for visual evaluation."""
        content = []
        try:
            # Sample 3 keyframes: beginning, middle, end
            duration = montage.actual_duration
            sample_times = [
                duration * 0.1,
                duration * 0.5,
                duration * 0.9,
            ]
            labels = ["Opening", "Middle", "Closing"]

            for i, (t, label) in enumerate(zip(sample_times, labels)):
                try:
                    thumb = extract_thumbnail(
                        montage.output_path,
                        t,
                        f"critic_keyframe_{i}",
                        width=480,
                    )
                    img_b64 = thumbnail_to_base64(thumb)
                    content.append({
                        "type": "text",
                        "text": f"\n**{label} frame** (at {t:.1f}s):"
                    })
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        }
                    })
                except Exception as e:
                    print(f"[CRITIC] Warning: could not extract keyframe at {t:.1f}s: {e}")

        except Exception as e:
            print(f"[CRITIC] Warning: visual evaluation unavailable: {e}")

        if not content:
            content = [{"type": "text", "text": "Visual frames not available for this evaluation."}]

        return content
