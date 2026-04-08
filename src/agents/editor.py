"""EDITOR agent: EditPlan → rendered video file."""
from __future__ import annotations

from src.models import EditPlan, MontageResult
from src.video.editor import execute_edit_plan


class EditorAgent:
    """No Claude call needed — executes the EditPlan deterministically via moviepy."""

    def run(self, plan: EditPlan, output_path: str) -> MontageResult:
        print(f"[EDITOR] Rendering {len(plan.edits)} clips → {output_path}")
        return execute_edit_plan(plan, output_path)
