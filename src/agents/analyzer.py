"""ANALYZER agent: rushes → VideoSegment list with technical + semantic metadata."""
from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel

from src.agents.base_agent import BaseAgent
from src.config import ANALYZER_MODEL, MAX_SEGMENTS_PER_RUSH, TARGET_MONTAGE_DURATION
from src.agents.annotator import SegmentSemantics  # contrat partagé avec le moteur graphe
from src.models import AnalysisResult, VideoSegment
from src.video.probe import probe_video, detect_scenes
from src.video.thumbnails import extract_thumbnail, build_vision_content


SYSTEM_PROMPT = """You are an expert video editor and cinematographer analyzing raw video rushes.

For each video segment shown, analyze:
1. Visual quality (0.0-1.0): focus, exposure, composition, stability
2. Semantic tags: what's happening (action, dialogue, establishing_shot, close_up, wide_shot, b_roll, interview, etc.)
3. Dominant emotion: energetic, calm, tense, joyful, melancholic, suspenseful, neutral
4. Suggested narrative role: opening, build_up, climax, resolution, outro, b_roll
5. Whether this segment is worth including in a final edit

Be critical and honest. Prioritize high-quality, visually interesting segments.
Return your analysis for ALL segments shown."""


class SemanticAnalysisResult(BaseModel):
    segments: list[SegmentSemantics]
    overall_summary: str
    recommended_output_duration: float


# Max images per Claude call to avoid context overflow
_BATCH_SIZE = 4


class AnalyzerAgent(BaseAgent):
    def __init__(self):
        super().__init__(model=ANALYZER_MODEL)

    def run(self, rushes_paths: list[str]) -> AnalysisResult:
        all_segments: list[VideoSegment] = []
        total_duration = 0.0
        segments_with_thumbs = []

        print(f"[ANALYZER] Processing {len(rushes_paths)} rush file(s)...")

        for rush_path in rushes_paths:
            print(f"[ANALYZER] Probing: {rush_path}")
            try:
                meta = probe_video(rush_path)
            except Exception as e:
                print(f"[ANALYZER] Warning: could not probe {rush_path}: {e}")
                continue

            total_duration += meta.duration

            try:
                scenes = detect_scenes(rush_path, threshold=0.25)
            except Exception as e:
                print(f"[ANALYZER] Warning: scene detection failed, using full file: {e}")
                scenes = [(0.0, meta.duration)]

            scenes = scenes[:MAX_SEGMENTS_PER_RUSH]
            print(f"[ANALYZER] Found {len(scenes)} scene(s) in {Path(rush_path).name}")

            for i, (start, end) in enumerate(scenes):
                duration = end - start
                thumb_name = f"{Path(rush_path).stem}_scene_{i:03d}"
                mid_time = start + duration * 0.3

                try:
                    thumb_path = extract_thumbnail(rush_path, mid_time, thumb_name)
                except Exception as e:
                    print(f"[ANALYZER] Warning: thumbnail extraction failed: {e}")
                    thumb_path = None

                segments_with_thumbs.append(({
                    "source_file": rush_path,
                    "start_time": start,
                    "end_time": end,
                    "duration": duration,
                }, thumb_path))

        if not segments_with_thumbs:
            raise RuntimeError("No segments could be extracted from the provided rush files")

        valid_pairs = [(s, t) for s, t in segments_with_thumbs if t is not None]
        if not valid_pairs:
            print("[ANALYZER] No thumbnails available, using basic analysis")
            return self._fallback_analysis(segments_with_thumbs, total_duration)

        # Analyze in batches to avoid context overflow
        all_semantics: list[SegmentSemantics] = []
        overall_summaries: list[str] = []

        for batch_start in range(0, len(valid_pairs), _BATCH_SIZE):
            batch = valid_pairs[batch_start: batch_start + _BATCH_SIZE]
            # Re-index within each batch call starting at 1
            batch_semantics = self._analyze_batch(batch, batch_start, total_duration)
            all_semantics.extend(batch_semantics)
            overall_summaries.append(f"Batch {batch_start // _BATCH_SIZE + 1}: {len(batch)} segments analyzed")

        # Garde d'alignement. Sans elle, un batch qui renvoie moins d'analyses
        # que d'images décale TOUS les segments suivants : la note et l'émotion
        # d'un plan sont attribuées à un autre, silencieusement. Le moteur graphe
        # (`--engine graph`) reprend le batch image par image ; ici on échoue.
        if len(all_semantics) != len(valid_pairs):
            raise RuntimeError(
                f"Désalignement analyses/segments : {len(all_semantics)} pour "
                f"{len(valid_pairs)} vignettes. Utiliser --engine graph, qui "
                f"reprend le batch unitairement au lieu de deviner l'appariement."
            )

        # Merge technical + semantic data
        for i, (seg_dict, thumb_path) in enumerate(valid_pairs):
            semantics = next((s for s in all_semantics if s.segment_index == i + 1), None)
            if semantics is None and i < len(all_semantics):
                semantics = all_semantics[i]

            all_segments.append(VideoSegment(
                source_file=seg_dict["source_file"],
                start_time=seg_dict["start_time"],
                end_time=seg_dict["end_time"],
                duration=seg_dict["duration"],
                quality_score=semantics.quality_score if semantics else 0.5,
                semantic_tags=semantics.semantic_tags if semantics else ["b_roll"],
                emotion=semantics.emotion if semantics else "neutral",
                suggested_role=semantics.suggested_role if semantics else "b_roll",
                thumbnail_path=thumb_path,
            ))

        recommended_duration = min(total_duration, TARGET_MONTAGE_DURATION)

        return AnalysisResult(
            segments=all_segments,
            total_rushes_duration=total_duration,
            recommended_output_duration=recommended_duration,
            summary=" | ".join(overall_summaries) if overall_summaries else "Analysis complete",
        )

    def _analyze_batch(
        self,
        batch: list[tuple[dict, str]],
        global_offset: int,
        total_duration: float,
    ) -> list[SegmentSemantics]:
        """Analyze a small batch of segments. Falls back to defaults on failure."""
        print(f"[ANALYZER] Analyzing batch of {len(batch)} segment(s) (offset={global_offset})...")

        vision_content = build_vision_content(batch)
        vision_content.append({
            "type": "text",
            "text": (
                f"\nAnalyze the {len(batch)} segment(s) shown above. "
                f"Number them starting at segment_index={global_offset + 1}. "
                f"Target final video duration: ~{TARGET_MONTAGE_DURATION}s. "
                f"Total rush duration: {total_duration:.1f}s."
            ),
        })

        try:
            result: SemanticAnalysisResult = self.call(
                system=SYSTEM_PROMPT,
                user_content=vision_content,
                output_schema=SemanticAnalysisResult,
                max_tokens=2048,
            )
            # Fix segment_index to be globally consistent
            for j, seg in enumerate(result.segments):
                seg.segment_index = global_offset + j + 1
            return result.segments
        except Exception as e:
            print(f"[ANALYZER] Warning: batch analysis failed ({e}), using defaults for this batch")
            return [
                SegmentSemantics(
                    segment_index=global_offset + j + 1,
                    quality_score=0.5,
                    semantic_tags=["b_roll"],
                    emotion="neutral",
                    suggested_role="b_roll",
                    include_recommendation=True,
                    notes="Auto-fallback (analysis failed)",
                )
                for j in range(len(batch))
            ]

    def _fallback_analysis(
        self, segments_with_thumbs: list, total_duration: float
    ) -> AnalysisResult:
        segments = [
            VideoSegment(
                source_file=s["source_file"],
                start_time=s["start_time"],
                end_time=s["end_time"],
                duration=s["duration"],
                quality_score=0.5,
                semantic_tags=["b_roll"],
                emotion="neutral",
                suggested_role="b_roll",
                thumbnail_path=None,
            )
            for s, _ in segments_with_thumbs
        ]
        return AnalysisResult(
            segments=segments,
            total_rushes_duration=total_duration,
            recommended_output_duration=min(total_duration, TARGET_MONTAGE_DURATION),
            summary="Fallback analysis — no visual inspection performed",
        )
