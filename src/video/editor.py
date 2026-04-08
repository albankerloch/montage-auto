"""moviepy-based video editor that executes an EditPlan."""
from __future__ import annotations
from pathlib import Path

from src.models import EditPlan, MontageResult


def execute_edit_plan(plan: EditPlan, output_path: str) -> MontageResult:
    """
    Execute the EditPlan using moviepy, write the final video to output_path.
    Returns a MontageResult with the actual output file info.
    """
    import imageio_ffmpeg
    import os
    # Point moviepy to the embedded ffmpeg binary
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", imageio_ffmpeg.get_ffmpeg_exe())

    from moviepy import VideoFileClip, concatenate_videoclips
    import moviepy.video.fx as vfx

    clips = []

    for edit in sorted(plan.edits, key=lambda e: e.order):
        seg = edit.segment
        try:
            clip = VideoFileClip(seg.source_file)
        except Exception as e:
            print(f"[EDITOR] Warning: could not load {seg.source_file}: {e}, skipping")
            continue

        # Clamp times to file bounds
        start = max(0.0, seg.start_time)
        end = min(clip.duration, seg.end_time)
        if end <= start:
            clip.close()
            continue

        clip = clip.subclipped(start, end)

        # Apply speed factor
        if edit.speed_factor != 1.0:
            clip = clip.with_effects([vfx.MultiplySpeed(edit.speed_factor)])

        # Apply audio level (moviepy 2.x uses MultiplyVolume effect)
        if clip.audio is not None and edit.audio_level != 1.0:
            import moviepy.audio.fx as afx
            clip = clip.with_effects([afx.MultiplyVolume(edit.audio_level)])

        clips.append(clip)

    if not clips:
        raise RuntimeError("No clips could be loaded from the edit plan")

    # Concatenate all clips
    final = concatenate_videoclips(clips, method="compose")

    # Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        logger=None,  # suppress verbose moviepy output
    )

    actual_duration = final.duration
    for c in clips:
        c.close()
    final.close()

    return MontageResult(
        output_path=output_path,
        actual_duration=actual_duration,
        edit_plan=plan,
    )
