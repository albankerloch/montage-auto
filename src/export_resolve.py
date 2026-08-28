"""
Build the final EditPlan directly as a timeline inside DaVinci Resolve,
via the official scripting API — no EDL/FCPXML import step needed.

Complements src/export.py (file-based exports) : here Resolve must be OPEN,
with an active project, and scripting enabled
(Preferences > System > General > "External scripting using" = Local).

Two timelines are created:
  - "<title>"          : the edit plan, in order
  - (nothing else — transitions/speed are reported, see limitations below)

Usage:
    # standalone, from a plan saved by export_all():
    python -m src.export_resolve output/<title>_plan.json

    # or wired into the pipeline:
    python -m src.main rushes/ --resolve

API limitations (documented, not silent):
  - AppendToTimeline cannot create dissolves/fades → transitions are listed
    in the console; apply them in Resolve (they are also present in the EDL
    and FCPXML exports).
  - Retime (speed_factor != 1.0) cannot be set through AppendToTimeline →
    clips are appended at 100% and flagged in the console.
  - Source in/out points are clamped to the real clip bounds reported by
    Resolve (anti-hallucination guard: an LLM-generated plan may reference
    times slightly outside the media).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from src.models import EditPlan


# ── Resolve connection ────────────────────────────────────────────────────────

def _get_resolve():
    """Import DaVinciResolveScript, trying RESOLVE_SCRIPT_API then the
    standard install locations for each OS."""
    candidates = []
    if os.environ.get("RESOLVE_SCRIPT_API"):
        candidates.append(Path(os.environ["RESOLVE_SCRIPT_API"]) / "Modules")
    candidates += [
        Path("/opt/resolve/Developer/Scripting/Modules"),
        Path("/Library/Application Support/Blackmagic Design/DaVinci Resolve"
             "/Developer/Scripting/Modules"),
        Path(os.environ.get("PROGRAMDATA", "C:/ProgramData"))
        / "Blackmagic Design/DaVinci Resolve/Support/Developer/Scripting/Modules",
    ]
    for c in candidates:
        if c.exists() and str(c) not in sys.path:
            sys.path.append(str(c))
    try:
        import DaVinciResolveScript as dvr  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "Cannot import DaVinciResolveScript. Run this from Resolve's Py3 "
            "console (Workspace > Console) or set RESOLVE_SCRIPT_API to the "
            "'Developer/Scripting' folder of your Resolve install."
        ) from e
    resolve = dvr.scriptapp("Resolve")
    if resolve is None:
        raise SystemExit(
            "Resolve is not responding: is it open, with external scripting "
            "set to 'Local' in Preferences > System > General?"
        )
    return resolve


# ── Timeline build ────────────────────────────────────────────────────────────

def build_timeline_in_resolve(plan: EditPlan, timeline_name: str | None = None) -> dict:
    """
    Import the plan's source media into the current project's media pool and
    append every edit to a new timeline. Returns a summary dict.
    """
    resolve = _get_resolve()
    project = resolve.GetProjectManager().GetCurrentProject()
    if project is None:
        raise SystemExit("No project open in Resolve.")
    media_pool = project.GetMediaPool()

    edits = sorted(plan.edits, key=lambda e: e.order)
    files = sorted({str(Path(e.segment.source_file).resolve()) for e in edits})
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        print(f"[RESOLVE] Warning: {len(missing)} source file(s) not found on "
              f"disk: {missing[:3]}…", file=sys.stderr)

    # 1) Import media (idempotent: Resolve skips duplicates), then index the
    #    whole bin tree path→MediaPoolItem, because already-present items are
    #    not returned by ImportMedia.
    media_pool.ImportMedia([f for f in files if f not in missing])
    items = _index_bins(media_pool.GetRootFolder())

    # 2) Create the timeline with a unique name.
    name = _unique_timeline_name(project, timeline_name or plan.title or "MONTAGE_AUTO")
    timeline = media_pool.CreateEmptyTimeline(name)
    if timeline is None:
        raise SystemExit(f"Failed to create timeline '{name}'.")
    project.SetCurrentTimeline(timeline)

    # 3) Append clips.
    appended, skipped, needs_retime, needs_transition = 0, [], [], []
    clip_infos = []
    for e in edits:
        seg = e.segment
        item = items.get(str(Path(seg.source_file).resolve()))
        if item is None:
            skipped.append(Path(seg.source_file).name)
            continue

        fps = _clip_fps(item)
        start_f = int(round(seg.start_time * fps))
        end_f = int(round(seg.end_time * fps)) - 1
        # Anti-hallucination clamp against the real clip length.
        total = _clip_frames(item)
        if total:
            start_f = max(0, min(start_f, total - 2))
            end_f = max(start_f + 1, min(end_f, total - 1))
        clip_infos.append({
            "mediaPoolItem": item,
            "startFrame": start_f,
            "endFrame": end_f,
        })
        appended += 1
        if e.speed_factor != 1.0:
            needs_retime.append(f"#{e.order} {Path(seg.source_file).name} "
                                f"×{e.speed_factor:g}")
        if e.transition_in != "cut" and e.transition_duration > 0:
            needs_transition.append(f"#{e.order} {e.transition_in} "
                                    f"{e.transition_duration:g}s")

    if clip_infos:
        media_pool.AppendToTimeline(clip_infos)

    summary = {
        "timeline": name,
        "appended": appended,
        "skipped": skipped,
        "manual_retimes": needs_retime,
        "manual_transitions": needs_transition,
    }
    print(f"[RESOLVE] Timeline '{name}': {appended} clip(s) appended.")
    if skipped:
        print(f"[RESOLVE] Skipped (media not found in pool): {skipped}")
    if needs_retime:
        print(f"[RESOLVE] Apply retime manually (API limitation): {needs_retime}")
    if needs_transition:
        print(f"[RESOLVE] Apply transitions manually (API limitation): "
              f"{needs_transition}")
    return summary


# ── Helpers ───────────────────────────────────────────────────────────────────

def _index_bins(folder, acc: dict | None = None) -> dict:
    acc = acc if acc is not None else {}
    for clip in folder.GetClipList() or []:
        p = clip.GetClipProperty("File Path")
        if p:
            acc[str(Path(p).resolve())] = clip
    for sub in folder.GetSubFolderList() or []:
        _index_bins(sub, acc)
    return acc


def _clip_fps(item, default: float = 25.0) -> float:
    try:
        return float(item.GetClipProperty("FPS")) or default
    except (TypeError, ValueError):
        return default


def _clip_frames(item) -> int | None:
    try:
        return int(item.GetClipProperty("Frames"))
    except (TypeError, ValueError):
        return None


def _unique_timeline_name(project, base: str) -> str:
    count = int(project.GetTimelineCount() or 0)
    existing = {project.GetTimelineByIndex(i + 1).GetName() for i in range(count)}
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


# ── Standalone CLI ────────────────────────────────────────────────────────────

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build an EditPlan as a timeline inside DaVinci Resolve "
                    "(Resolve must be open).")
    parser.add_argument("plan_json",
                        help="Path to a plan JSON written by export_all() "
                             "(output/<title>_plan.json)")
    parser.add_argument("--name", default=None, help="Timeline name override")
    args = parser.parse_args()

    plan = EditPlan.model_validate(
        json.loads(Path(args.plan_json).read_text(encoding="utf-8")))
    build_timeline_in_resolve(plan, timeline_name=args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
