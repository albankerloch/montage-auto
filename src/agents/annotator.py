"""ANNOTATOR : rushes → sémantique par scène. Remplace la partie vision d'ANALYZER.

Deux corrections de fond par rapport à `analyzer.py` :

1. **Alignement strict.** L'ancien code appariait analyses et segments via
   `next(s for s in all_semantics if s.segment_index == i + 1)` puis se rabattait
   sur `all_semantics[i]`. Si un batch renvoyait 3 analyses pour 4 images, tous
   les segments suivants étaient décalés — la note et l'émotion d'un plan
   attribuées à un autre, silencieusement. Ici, un batch dont le compte ne
   correspond pas est repassé image par image, ce qui rend le décalage
   impossible par construction.

2. **Échec explicite.** L'ancien fallback écrivait `quality_score=0.5,
   emotion="neutral"` : indiscernable en aval d'une vraie analyse. Ici l'échec
   est porté par le contrat (`failed=True`) et le segment est écarté du pool de
   candidats plutôt que maquillé en donnée valide.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent
from src.config import ANALYZER_MODEL, ANNOTATE_BATCH_SIZE
from src.video.thumbnails import build_vision_content

SYSTEM_PROMPT = """You are an expert video editor and cinematographer analyzing raw video rushes.

For each numbered frame, return one analysis, in the same order, with the same count.

1. quality_score (0.0-1.0): focus, exposure, composition, framing
2. semantic_tags: what is happening
3. emotion: dominant emotion
4. suggested_role: narrative role in an edit
5. include_recommendation: worth including at all

Be critical and honest. Return EXACTLY one entry per frame shown, in order."""


class SegmentSemantics(BaseModel):
    segment_index: int
    quality_score: float = Field(ge=0.0, le=1.0)
    semantic_tags: list[str]
    emotion: str
    suggested_role: str
    include_recommendation: bool
    notes: str = ""
    failed: bool = False


class BatchAnalysis(BaseModel):
    segments: list[SegmentSemantics]


def _failed(index: int, why: str) -> SegmentSemantics:
    return SegmentSemantics(
        segment_index=index,
        quality_score=0.0,
        semantic_tags=[],
        emotion="unknown",
        suggested_role="unknown",
        include_recommendation=False,
        notes=why,
        failed=True,
    )


class AnnotatorAgent(BaseAgent):
    def __init__(self, model: str = ANALYZER_MODEL):
        super().__init__(model=model)

    def annotate(
        self,
        rush_path: str,
        scenes: list[tuple[float, float]],
        thumbs: list[str],
        batch_size: int = ANNOTATE_BATCH_SIZE,
    ) -> list[SegmentSemantics]:
        """Renvoie exactement `len(scenes)` analyses, dans l'ordre des scènes."""
        if len(thumbs) != len(scenes):
            raise ValueError(
                f"{Path(rush_path).name}: {len(thumbs)} vignettes pour {len(scenes)} scènes"
            )

        out: list[SegmentSemantics] = []
        for start in range(0, len(scenes), batch_size):
            chunk = list(zip(scenes[start : start + batch_size], thumbs[start : start + batch_size]))
            out.extend(self._batch(rush_path, chunk, start))

        assert len(out) == len(scenes), "invariant d'alignement rompu"
        return out

    def _batch(
        self, rush_path: str, chunk: list[tuple[tuple[float, float], str]], offset: int
    ) -> list[SegmentSemantics]:
        pairs = [
            (
                {
                    "source_file": rush_path,
                    "start_time": s,
                    "end_time": e,
                    "duration": e - s,
                },
                thumb,
            )
            for (s, e), thumb in chunk
        ]

        try:
            result: BatchAnalysis = self.call(
                system=SYSTEM_PROMPT,
                user_content=build_vision_content(pairs)
                + [
                    {
                        "type": "text",
                        "text": (
                            f"Renvoie exactement {len(pairs)} analyses, "
                            "une par image, dans l'ordre."
                        ),
                    }
                ],
                output_schema=BatchAnalysis,
                max_tokens=2048,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[ANNOTATOR] batch offset={offset} en échec ({e})")
            return self._one_by_one(rush_path, chunk, offset, why=f"batch failed: {e}")

        if len(result.segments) != len(pairs):
            # Le décalage silencieux se jouait exactement ici. On ne devine pas
            # l'appariement : on repasse image par image, l'alignement redevient
            # trivialement vrai.
            print(
                f"[ANNOTATOR] batch offset={offset}: {len(result.segments)} analyses "
                f"pour {len(pairs)} images → reprise unitaire"
            )
            return self._one_by_one(rush_path, chunk, offset, why="count mismatch")

        for j, seg in enumerate(result.segments):
            seg.segment_index = offset + j
            seg.failed = False
        return result.segments

    def _one_by_one(
        self,
        rush_path: str,
        chunk: list[tuple[tuple[float, float], str]],
        offset: int,
        why: str,
    ) -> list[SegmentSemantics]:
        out: list[SegmentSemantics] = []
        for j, ((s, e), thumb) in enumerate(chunk):
            index = offset + j
            pair = [({"source_file": rush_path, "start_time": s, "end_time": e, "duration": e - s}, thumb)]
            try:
                single: BatchAnalysis = self.call(
                    system=SYSTEM_PROMPT,
                    user_content=build_vision_content(pair)
                    + [{"type": "text", "text": "Renvoie exactement 1 analyse."}],
                    output_schema=BatchAnalysis,
                    max_tokens=512,
                )
                if len(single.segments) != 1:
                    out.append(_failed(index, f"{why}; unitaire: {len(single.segments)} analyses"))
                    continue
                seg = single.segments[0]
                seg.segment_index = index
                seg.failed = False
                out.append(seg)
            except Exception as e2:  # noqa: BLE001
                print(f"[ANNOTATOR] scène {index} non annotée: {e2}")
                out.append(_failed(index, f"{why}; unitaire: {e2}"))
        return out
