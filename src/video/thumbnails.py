"""Extract thumbnail frames from video segments for Claude vision analysis."""
from __future__ import annotations
import base64
import subprocess
from pathlib import Path

from src.config import THUMBNAILS_DIR


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_thumbnail(
    file_path: str,
    timestamp: float,
    output_name: str,
    width: int = 640,
) -> str:
    """Extract a single frame at `timestamp` seconds, return saved path."""
    out_path = THUMBNAILS_DIR / f"{output_name}.jpg"
    cmd = [
        _ffmpeg_exe(), "-y",
        "-ss", str(timestamp),
        "-i", file_path,
        "-vframes", "1",
        "-vf", f"scale={width}:-1",
        "-q:v", "3",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"Thumbnail extraction failed: {result.stderr.decode(errors='replace')}")
    return str(out_path)


def thumbnail_to_base64(thumb_path: str) -> str:
    """Return base64-encoded JPEG content."""
    with open(thumb_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode()


def build_vision_content(
    segments_with_thumbs: list[tuple[dict, str]],
) -> list[dict]:
    """
    Build the content blocks list for Claude vision.
    segments_with_thumbs: list of (segment_dict, thumbnail_path)
    """
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "You are analyzing video segments. For each numbered frame below, "
                "provide your assessment of that specific segment. "
                "The frames are extracted from different rush video files."
            ),
        }
    ]

    for i, (seg, thumb_path) in enumerate(segments_with_thumbs):
        img_b64 = thumbnail_to_base64(thumb_path)
        content.append(
            {
                "type": "text",
                "text": (
                    f"\n--- Segment {i + 1} ---\n"
                    f"File: {seg['source_file']}\n"
                    f"Time: {seg['start_time']:.1f}s -> {seg['end_time']:.1f}s "
                    f"(duration: {seg['duration']:.1f}s)"
                ),
            }
        )
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_b64,
                },
            }
        )

    return content
