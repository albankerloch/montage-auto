"""Video metadata extraction and scene detection using embedded ffmpeg."""
from __future__ import annotations
import json
import shutil
import subprocess
from pathlib import Path

from src.models import VideoMetadata


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _ffprobe_cmd(file_path: str) -> list[str] | None:
    """Commande ffprobe si un ffprobe est réellement disponible, sinon None.

    ATTENTION : la version précédente prétendait que « ffmpeg expose la sortie
    de type probe via les mêmes flags -show_streams / -print_format ». C'est
    faux — ffmpeg répond `Unrecognized option 'print_format'`. Or imageio-ffmpeg
    n'embarque QUE ffmpeg (c'est tout l'argument du « pas besoin d'installer
    ffmpeg »), donc sur une installation propre cette branche échouait toujours
    et chaque probe retombait en silence sur moviepy, qui ouvre un
    VideoFileClip complet juste pour lire une durée. On préfère désormais les
    métadonnées d'imageio-ffmpeg, qui ne demandent aucun ffprobe.
    """
    ffmpeg = _ffmpeg_exe()
    for candidate in (
        Path(ffmpeg).parent / "ffprobe.exe",
        Path(ffmpeg).parent / "ffprobe",
        Path(shutil.which("ffprobe") or "/nonexistent"),
    ):
        if candidate.exists():
            return [
                str(candidate), "-v", "quiet", "-print_format", "json",
                "-show_streams", "-show_format", file_path,
            ]
    return None


def _probe_with_imageio(file_path: str) -> VideoMetadata:
    """Métadonnées via imageio-ffmpeg : pas de ffprobe, pas de moviepy."""
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(file_path)
    try:
        meta = next(reader)
    finally:
        reader.close()

    width, height = meta.get("size", (0, 0))
    return VideoMetadata(
        file_path=file_path,
        duration=float(meta.get("duration", 0.0) or 0.0),
        fps=float(meta.get("fps", 25.0) or 25.0),
        width=int(width),
        height=int(height),
        has_audio=meta.get("audio_codec") is not None,
        codec=str(meta.get("codec", "unknown")),
        file_size_bytes=Path(file_path).stat().st_size,
    )


def probe_video(file_path: str) -> VideoMetadata:
    """Extract technical metadata from a video file."""
    cmd = _ffprobe_cmd(file_path)
    if cmd is None:
        return _probe_with_imageio(file_path)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0 or not result.stdout.strip():
        return _probe_with_imageio(file_path)

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return _probe_with_imageio(file_path)

    video_stream = next((s for s in data.get("streams", []) if s["codec_type"] == "video"), None)
    audio_stream = next((s for s in data.get("streams", []) if s["codec_type"] == "audio"), None)
    fmt = data.get("format", {})

    if not video_stream:
        return _probe_with_imageio(file_path)

    fps_raw = video_stream.get("r_frame_rate", "25/1")
    num, den = fps_raw.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 25.0

    duration = float(video_stream.get("duration", 0) or fmt.get("duration", 0) or 0)
    file_size = int(fmt.get("size", 0) or Path(file_path).stat().st_size)

    return VideoMetadata(
        file_path=file_path,
        duration=duration,
        fps=fps,
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        has_audio=audio_stream is not None,
        codec=video_stream.get("codec_name", "unknown"),
        file_size_bytes=file_size,
    )


def _probe_with_moviepy(file_path: str) -> VideoMetadata:
    """Dernier recours historique. Conservé, mais plus sur le chemin nominal."""
    import os
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", _ffmpeg_exe())
    from moviepy import VideoFileClip

    clip = VideoFileClip(file_path)
    meta = VideoMetadata(
        file_path=file_path,
        duration=clip.duration,
        fps=clip.fps,
        width=clip.size[0],
        height=clip.size[1],
        has_audio=clip.audio is not None,
        codec="unknown",
        file_size_bytes=Path(file_path).stat().st_size,
    )
    clip.close()
    return meta


def detect_scenes(file_path: str, threshold: float = 0.3) -> list[tuple[float, float]]:
    """
    Detect scene boundaries using ffmpeg scene detection filter.
    Returns list of (start, end) tuples in seconds.
    """
    import os
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", _ffmpeg_exe())

    # Get duration
    try:
        meta = probe_video(file_path)
        duration = meta.duration
    except Exception:
        duration = 0.0

    ffmpeg = _ffmpeg_exe()
    cmd = [
        ffmpeg, "-i", file_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    scene_times = [0.0]
    for line in result.stderr.split("\n"):
        if "pts_time:" in line and "showinfo" in line:
            try:
                pts_part = line.split("pts_time:")[1].split()[0]
                t = float(pts_part)
                if t > 0.5:
                    scene_times.append(t)
            except (IndexError, ValueError):
                continue

    if duration > 0:
        scene_times.append(duration)

    scenes = []
    for i in range(len(scene_times) - 1):
        start = scene_times[i]
        end = scene_times[i + 1]
        if end - start >= 0.5:
            scenes.append((start, end))

    if not scenes and duration > 0:
        scenes = [(0.0, duration)]

    return scenes
