"""Non-linear state machine orchestrator for the video montage pipeline."""
from __future__ import annotations
import traceback
from uuid import uuid4

from src.agents.analyzer import AnalyzerAgent
from src.agents.critic import CriticAgent
from src.agents.editor import EditorAgent
from src.agents.quality import QualityAgent
from src.agents.revision import RevisionAgent
from src.agents.scenario import ScenarioAgent
from src.config import MAX_ITERATIONS, OUTPUT_DIR, QUALITY_THRESHOLD
from src.models import OrchestrationState


class Orchestrator:
    """
    Non-linear state machine orchestrator.

    Flow:
        START → ANALYZE → SCENARIO → EDIT → CRITIC
                              ▲                │
                              │          score < threshold
                          REVISION ◄──────────┘
                              │
                         (loop back)
                              │
                          score OK or max_iter
                              │
                           QUALITY → END
    """

    def __init__(self, max_iterations: int = MAX_ITERATIONS):
        self.max_iterations = max_iterations
        self.analyzer = AnalyzerAgent()
        self.scenario = ScenarioAgent()
        self.editor = EditorAgent()
        self.critic = CriticAgent()
        self.revision = RevisionAgent()
        self.quality = QualityAgent()

    def run(
        self,
        rushes_paths: list[str],
        progress_callback=None,
    ) -> OrchestrationState:
        """
        Run the full orchestration pipeline.

        Args:
            rushes_paths: List of video file paths to process
            progress_callback: Optional callable(node, message) for UI updates
        """
        state = OrchestrationState(
            run_id=str(uuid4()),
            current_node="START",
            iteration=0,
            max_iterations=self.max_iterations,
            rushes_paths=rushes_paths,
        )

        def notify(node: str, message: str):
            state.log(node, message)
            if progress_callback:
                progress_callback(node, message)

        notify("START", f"Starting orchestration with {len(rushes_paths)} rush file(s)")

        while state.current_node != "END":
            node = state.current_node
            try:
                state = self._transition(state, notify)
            except Exception as e:
                error_msg = f"Error in node {node}: {e}\n{traceback.format_exc()}"
                state.error = error_msg
                notify(node, f"ERROR: {e}")
                state.current_node = "END"

        if state.final_output_path:
            notify("END", f"Pipeline complete. Output: {state.final_output_path}")
        else:
            notify("END", f"Pipeline ended. Error: {state.error or 'unknown'}")

        return state

    def _transition(self, state: OrchestrationState, notify) -> OrchestrationState:
        node = state.current_node

        # ─── START ───────────────────────────────────────────────────────────
        if node == "START":
            state.current_node = "ANALYZE"

        # ─── ANALYZE ─────────────────────────────────────────────────────────
        elif node == "ANALYZE":
            notify("ANALYZE", "Extracting segments and analyzing rushes...")
            state.analysis = self.analyzer.run(state.rushes_paths)
            notify(
                "ANALYZE",
                f"Found {len(state.analysis.segments)} segments, "
                f"total rush duration: {state.analysis.total_rushes_duration:.1f}s, "
                f"recommended output: {state.analysis.recommended_output_duration:.1f}s",
            )
            state.current_node = "SCENARIO"

        # ─── SCENARIO ────────────────────────────────────────────────────────
        elif node == "SCENARIO":
            if state.revision_instructions:
                notify("SCENARIO", "Generating revised edit plan based on critic feedback...")
            else:
                notify("SCENARIO", "Generating initial edit plan...")

            state.edit_plan = self.scenario.run(
                state.analysis,
                state.revision_instructions,
            )
            state.revision_instructions = None  # consumed

            notify(
                "SCENARIO",
                f"Edit plan: '{state.edit_plan.title}' — "
                f"{len(state.edit_plan.edits)} cuts, "
                f"{state.edit_plan.total_duration:.1f}s planned",
            )
            state.current_node = "EDIT"

        # ─── EDIT ─────────────────────────────────────────────────────────────
        elif node == "EDIT":
            output_path = str(OUTPUT_DIR / f"montage_iter_{state.iteration}.mp4")
            notify("EDIT", f"Rendering video → {output_path}")
            state.montage = self.editor.run(state.edit_plan, output_path)
            notify("EDIT", f"Rendered {state.montage.actual_duration:.1f}s of video")
            state.current_node = "CRITIC"

        # ─── CRITIC ──────────────────────────────────────────────────────────
        elif node == "CRITIC":
            notify("CRITIC", "Evaluating montage quality...")
            state.critic_feedback = self.critic.run(state.montage, state.edit_plan)

            score = state.critic_feedback.score
            passed = state.critic_feedback.passed
            notify(
                "CRITIC",
                f"Score: {score:.2f} / 1.00 ({'✓ PASSED' if passed else '✗ FAILED — needs revision'}). "
                f"Iteration {state.iteration + 1}/{state.max_iterations}",
            )

            if passed or state.iteration >= state.max_iterations - 1:
                # Accept the current montage (either passed or we hit the limit)
                if not passed:
                    notify("CRITIC", f"Max iterations reached ({state.max_iterations}). Accepting best result.")
                state.current_node = "QUALITY"
            else:
                state.iteration += 1
                state.current_node = "REVISION"

        # ─── REVISION ────────────────────────────────────────────────────────
        elif node == "REVISION":
            notify(
                "REVISION",
                f"Generating revision instructions (iteration {state.iteration}/{state.max_iterations})...",
            )
            state.revision_instructions = self.revision.run(
                state.critic_feedback,
                state.edit_plan,
            )
            priority = state.revision_instructions.priority_changes
            notify("REVISION", f"Priority changes: {'; '.join(priority[:3])}")
            state.current_node = "SCENARIO"  # ← THE LOOP BACK

        # ─── QUALITY ─────────────────────────────────────────────────────────
        elif node == "QUALITY":
            notify("QUALITY", "Running final quality check...")
            quality_result = self.quality.run(state.montage)

            if quality_result.passed:
                notify("QUALITY", f"Quality check passed: {quality_result.recommendation}")
            else:
                notify(
                    "QUALITY",
                    f"Quality issues found: {', '.join(quality_result.issues)}. "
                    "Outputting best available result anyway (POC behavior).",
                )

            state.final_output_path = state.montage.output_path

            # Export EDL + FCPXML for DaVinci Resolve
            try:
                from src.export import export_all
                exports = export_all(state.edit_plan, str(OUTPUT_DIR))
                state.exports = exports
                notify("QUALITY", f"Exported: {', '.join(exports.keys())} → {OUTPUT_DIR}")
            except Exception as e:
                notify("QUALITY", f"Warning: export failed: {e}")

            state.current_node = "END"

        return state
