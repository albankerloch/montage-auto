from __future__ import annotations
from pathlib import Path

from pydantic import BaseModel, Field


def segment_key(source_file: str, start_time: float) -> str:
    """Identifiant d'un plan, tel qu'il s'affiche dans les rapports.

    C'est la poignée que le monteur recopie pour bannir ou imposer un plan. Elle
    doit donc être exactement ce qu'il lit à l'écran : même nom, même précision.
    La première version utilisait le chemin complet à 3 décimales tandis que le
    rapport affichait le nom de base à 2 décimales — et, pire, la borne
    *recadrée* plutôt que celle du segment. Trois écarts, donc un veto qui ne
    pouvait rien matcher.
    """
    return f"{Path(source_file).stem}@{start_time:.3f}"


class VideoMetadata(BaseModel):
    file_path: str
    duration: float
    fps: float
    width: int
    height: int
    has_audio: bool
    codec: str
    file_size_bytes: int


class VideoSegment(BaseModel):
    source_file: str
    start_time: float
    end_time: float
    duration: float
    quality_score: float = Field(ge=0.0, le=1.0, description="Technical/aesthetic quality")
    semantic_tags: list[str] = Field(description="e.g. action, dialogue, establishing_shot")
    emotion: str = Field(description="Dominant emotion: energetic, calm, tense, joyful, etc.")
    suggested_role: str = Field(description="opening, build_up, climax, resolution, outro, b_roll")
    thumbnail_path: str | None = None
    source_key: str | None = Field(
        default=None,
        description="Clé du segment d'origine, avant recadrage. C'est la poignée "
                    "de veto : sans elle, le plan exporté porte les bornes "
                    "recadrées et le monteur ne peut plus désigner le plan source.",
    )


class AnalysisResult(BaseModel):
    segments: list[VideoSegment]
    total_rushes_duration: float
    recommended_output_duration: float
    summary: str


class TimelineEdit(BaseModel):
    order: int
    segment: VideoSegment
    transition_in: str = Field(description="cut, fade, dissolve")
    transition_duration: float = Field(description="seconds, 0 for hard cut")
    audio_level: float = Field(ge=0.0, le=1.0)
    speed_factor: float = Field(description="1.0=normal, 2.0=double speed")


class EditPlan(BaseModel):
    title: str
    total_duration: float
    narrative_arc: str
    edits: list[TimelineEdit]
    music_suggestion: str | None = None


class MontageResult(BaseModel):
    output_path: str
    actual_duration: float
    edit_plan: EditPlan


class CriticFeedback(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    pacing_notes: str
    narrative_notes: str
    technical_notes: str
    specific_improvements: list[str]


class RevisionInstructions(BaseModel):
    priority_changes: list[str]
    segments_to_remove: list[str] = Field(description="source_file references to exclude")
    segments_to_emphasize: list[str] = Field(description="source_file references to keep/extend")
    pacing_adjustments: str | list[str]

    @property
    def pacing_adjustments_str(self) -> str:
        if isinstance(self.pacing_adjustments, list):
            return " ".join(self.pacing_adjustments)
        return self.pacing_adjustments


class OrchestrationState(BaseModel):
    run_id: str
    current_node: str = "START"
    iteration: int = 0
    max_iterations: int = 3
    rushes_paths: list[str]
    analysis: AnalysisResult | None = None
    edit_plan: EditPlan | None = None
    montage: MontageResult | None = None
    critic_feedback: CriticFeedback | None = None
    revision_instructions: RevisionInstructions | None = None
    history: list[dict] = Field(default_factory=list)
    final_output_path: str | None = None
    exports: dict[str, str] = Field(default_factory=dict, description="format -> file path")
    error: str | None = None

    def log(self, node: str, message: str) -> None:
        self.history.append({"node": node, "iteration": self.iteration, "message": message})
        print(f"[{node} | iter={self.iteration}] {message}")
