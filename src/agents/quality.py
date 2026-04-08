"""QUALITY agent: final gate check before output."""
from __future__ import annotations
import os
from pathlib import Path
from pydantic import BaseModel

from src.agents.base_agent import BaseAgent
from src.config import QUALITY_MODEL
from src.models import MontageResult


SYSTEM_PROMPT = """You are a quality control specialist for video production.
Perform a final quality assessment of the completed video montage.

Check:
1. Does the output seem complete (has a beginning, middle, end)?
2. Is the duration reasonable for the content?
3. Are there any obvious issues with the edit plan structure?

This is the final gate — be thorough but not overly strict.
A montage that passed the critic review should generally pass quality control."""


class QualityCheckResult(BaseModel):
    passed: bool
    issues: list[str]
    recommendation: str


class QualityAgent(BaseAgent):
    def __init__(self):
        super().__init__(model=QUALITY_MODEL)

    def run(self, montage: MontageResult) -> QualityCheckResult:
        # Technical checks first (no Claude needed)
        issues = []

        if not Path(montage.output_path).exists():
            issues.append(f"Output file does not exist: {montage.output_path}")

        file_size = Path(montage.output_path).stat().st_size if Path(montage.output_path).exists() else 0
        if file_size < 1000:
            issues.append(f"Output file is suspiciously small: {file_size} bytes")

        if montage.actual_duration < 1.0:
            issues.append(f"Output duration is too short: {montage.actual_duration:.1f}s")

        if len(montage.edit_plan.edits) == 0:
            issues.append("Edit plan has no cuts")

        # If technical checks passed, do a quick Claude review of the plan
        if not issues:
            print("[QUALITY] Running final quality check...")
            try:
                result = self.call(
                    system=SYSTEM_PROMPT,
                    user_content=(
                        f"Final montage stats:\n"
                        f"- Output: {montage.output_path}\n"
                        f"- Duration: {montage.actual_duration:.1f}s\n"
                        f"- Cuts: {len(montage.edit_plan.edits)}\n"
                        f"- Title: {montage.edit_plan.title}\n"
                        f"- Narrative arc: {montage.edit_plan.narrative_arc}\n\n"
                        "Does this montage pass final quality control?"
                    ),
                    output_schema=QualityCheckResult,
                    max_tokens=512,
                )
                return result
            except Exception as e:
                print(f"[QUALITY] Warning: Claude check failed: {e}, using technical result")

        passed = len(issues) == 0
        return QualityCheckResult(
            passed=passed,
            issues=issues,
            recommendation="Ready for output" if passed else "Fix technical issues before output",
        )
