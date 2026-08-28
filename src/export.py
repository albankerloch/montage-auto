"""
Export the final edit plan to NLE-compatible formats for DaVinci Resolve.

Supported formats:
  - EDL  (CMX 3600) — universal, works in every NLE
  - FCPXML (v1.10)  — Final Cut Pro XML, importable in DaVinci Resolve

Both formats reference the *original source files* so the NLE can re-link them
and do a proper conform at full quality (no re-encode).
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

from src.models import EditPlan, TimelineEdit


# ── Timecode helpers ──────────────────────────────────────────────────────────

def _seconds_to_timecode(seconds: float, fps: float = 25.0) -> str:
    """Convert seconds to SMPTE timecode HH:MM:SS:FF."""
    total_frames = int(round(seconds * fps))
    ff = total_frames % int(fps)
    ss = (total_frames // int(fps)) % 60
    mm = (total_frames // int(fps) // 60) % 60
    hh = total_frames // int(fps) // 3600
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def _tc_to_frames(seconds: float, fps: float = 25.0) -> int:
    return int(round(seconds * fps))


# ── EDL export ────────────────────────────────────────────────────────────────

def export_edl(plan: EditPlan, output_path: str, fps: float = 25.0) -> str:
    """
    Generate a CMX 3600 EDL file from an EditPlan.

    EDL format:
        TITLE: <title>
        FCM: NON-DROP FRAME

        001  <reel>  V     C        <src_in> <src_out> <rec_in> <rec_out>
        * FROM CLIP NAME: <filename>
    """
    lines: list[str] = []
    lines.append(f"TITLE: {plan.title}")
    lines.append("FCM: NON-DROP FRAME")
    lines.append("")

    record_tc = 0.0  # running record-side timecode in seconds

    for edit in sorted(plan.edits, key=lambda e: e.order):
        seg = edit.segment
        event_num = edit.order

        src_in = seg.start_time
        src_out = seg.end_time
        rec_in = record_tc
        rec_out = record_tc + seg.duration / edit.speed_factor

        # Reel name: max 8 chars, alphanumeric
        reel = Path(seg.source_file).stem[:8].upper().replace(" ", "_")

        # Transition
        if edit.transition_in == "cut" or edit.transition_duration == 0:
            trans = "C"
            trans_suffix = ""
        elif edit.transition_in == "dissolve":
            frames = max(1, _tc_to_frames(edit.transition_duration, fps))
            trans = "D"
            trans_suffix = f" {frames:04d}"
        else:  # fade
            frames = max(1, _tc_to_frames(edit.transition_duration, fps))
            trans = "W001"
            trans_suffix = f" {frames:04d}"

        src_in_tc  = _seconds_to_timecode(src_in, fps)
        src_out_tc = _seconds_to_timecode(src_out, fps)
        rec_in_tc  = _seconds_to_timecode(rec_in, fps)
        rec_out_tc = _seconds_to_timecode(rec_out, fps)

        lines.append(
            f"{event_num:03d}  {reel:<8}  V     {trans}{trans_suffix}"
            f"        {src_in_tc} {src_out_tc} {rec_in_tc} {rec_out_tc}"
        )
        lines.append(f"* FROM CLIP NAME: {Path(seg.source_file).name}")

        # Audio track if present
        lines.append(
            f"{event_num:03d}  {reel:<8}  A     {trans}{trans_suffix}"
            f"        {src_in_tc} {src_out_tc} {rec_in_tc} {rec_out_tc}"
        )

        # Speed remark
        if edit.speed_factor != 1.0:
            lines.append(f"* SPEED: {edit.speed_factor:.2f}x")

        lines.append("")
        record_tc = rec_out

    content = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(content, encoding="utf-8")
    return output_path


# ── FCPXML export ─────────────────────────────────────────────────────────────

def export_fcpxml(plan: EditPlan, output_path: str, fps: float = 25.0) -> str:
    """
    Generate an FCPXML 1.10 file from an EditPlan.
    DaVinci Resolve can import FCPXML via File > Import > Timeline.
    """
    fps_int = int(fps)
    # FCPXML uses rational frame durations: 1/fps
    frame_dur = f"1/{fps_int}s"

    def secs_to_rational(s: float) -> str:
        """Convert seconds to FCPXML rational time string e.g. '50/25s'."""
        frames = int(round(s * fps))
        return f"{frames}/{fps_int}s" if frames % fps_int != 0 else f"{frames // fps_int}s"

    # Root
    root = ET.Element("fcpxml", version="1.10")

    # Resources
    resources = ET.SubElement(root, "resources")

    # Collect unique source files
    source_files: dict[str, str] = {}  # path -> asset id
    for i, edit in enumerate(sorted(plan.edits, key=lambda e: e.order)):
        src = edit.segment.source_file
        if src not in source_files:
            asset_id = f"r{len(source_files) + 1}"
            source_files[src] = asset_id

            asset = ET.SubElement(resources, "asset",
                id=asset_id,
                name=Path(src).stem,
                src=Path(src).as_uri(),
                start="0s",
                duration=secs_to_rational(edit.segment.end_time + 1),
                hasVideo="1",
                hasAudio="1",
            )
            # Media format
            ET.SubElement(asset, "media-rep",
                kind="original-media",
                src=Path(src).as_uri(),
            )

    # Format resource (sequence settings)
    fmt = ET.SubElement(resources, "format",
        id="r0",
        name=f"FFVideoFormat{fps_int}p",
        frameDuration=frame_dur,
        width="1920",
        height="1080",
    )

    # Library > Event > Project > Sequence
    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name="Auto Montage")
    project = ET.SubElement(event, "project", name=plan.title)

    total_dur = secs_to_rational(plan.total_duration)
    sequence = ET.SubElement(project, "sequence",
        duration=total_dur,
        format="r0",
        tcStart="0s",
        tcFormat="NDF",
    )
    spine = ET.SubElement(sequence, "spine")

    record_frames = 0  # running record offset in frames

    for edit in sorted(plan.edits, key=lambda e: e.order):
        seg = edit.segment
        asset_id = source_files[seg.source_file]

        src_in_frames  = int(round(seg.start_time * fps))
        src_out_frames = int(round(seg.end_time * fps))
        clip_frames    = src_out_frames - src_in_frames

        offset = f"{record_frames}/{fps_int}s"
        duration = f"{clip_frames}/{fps_int}s"
        start    = f"{src_in_frames}/{fps_int}s"

        clip_el = ET.SubElement(spine, "clip",
            name=Path(seg.source_file).name,
            ref=asset_id,
            offset=offset,
            duration=duration,
            start=start,
        )

        # Speed effect
        if edit.speed_factor != 1.0:
            timeMap = ET.SubElement(clip_el, "timeMap")
            ET.SubElement(timeMap, "timept",
                time="0s",
                value="0s",
                interp="linear",
            )
            end_out = clip_frames / edit.speed_factor
            ET.SubElement(timeMap, "timept",
                time=f"{clip_frames}/{fps_int}s",
                value=f"{end_out:.6f}s",
                interp="linear",
            )

        # Audio role / volume
        ET.SubElement(clip_el, "audio-channel-source",
            srcCh="1, 2",
            role="dialogue",
        )

        # Transition
        if edit.transition_in in ("dissolve", "fade") and edit.transition_duration > 0:
            trans_frames = int(round(edit.transition_duration * fps))
            ET.SubElement(clip_el, "transition",
                name="Cross Dissolve" if edit.transition_in == "dissolve" else "Fade to Black",
                duration=f"{trans_frames}/{fps_int}s",
                offset=offset,
            )

        # Marker with narrative role
        ET.SubElement(clip_el, "marker",
            start="0s",
            duration=frame_dur,
            value=f"{seg.suggested_role} | {seg.emotion}",
        )

        record_frames += clip_frames

    # Pretty-print
    xml_str = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(indent="  ")
    # Remove the extra XML declaration minidom adds
    lines = xml_str.split("\n")
    if lines[0].startswith("<?xml"):
        lines[0] = '<?xml version="1.0" encoding="UTF-8"?>'
    content = "\n".join(lines)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(content, encoding="utf-8")
    return output_path


# ── Convenience: export both ──────────────────────────────────────────────────

def export_all(plan: EditPlan, output_dir: str, fps: float = 25.0) -> dict[str, str]:
    """Export EDL, FCPXML and the plan itself as JSON.
    Returns dict of format -> file path.

    The plan JSON is the input of `python -m src.export_resolve`, which builds
    the timeline directly inside DaVinci Resolve through the scripting API.
    """
    base = Path(output_dir) / _safe_name(plan.title)
    results = {}
    results["edl"]    = export_edl(plan, str(base.with_suffix(".edl")), fps)
    results["fcpxml"] = export_fcpxml(plan, str(base.with_suffix(".fcpxml")), fps)

    plan_path = base.parent / f"{base.name}_plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    results["plan"] = str(plan_path)
    return results


def _safe_name(s: str) -> str:
    """Sanitize string for use as a filename."""
    return "".join(c if c.isalnum() or c in "-_ " else "_" for c in s).strip().replace(" ", "_")
