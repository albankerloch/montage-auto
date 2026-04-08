"""CLI entry point for the video montage pipeline."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Automatic video montage using AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main rushes/clip1.mp4 rushes/clip2.mp4
  python -m src.main rushes/*.mp4 --max-iter 2
  python -m src.main rushes/ --output output/my_montage.mp4
        """,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Video files or a directory containing video files",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=3,
        help="Maximum critic→revision→scenario iterations (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path (default: output/final_montage.mp4)",
    )
    parser.add_argument(
        "--dump-state",
        action="store_true",
        help="Dump the final orchestration state to JSON",
    )

    args = parser.parse_args()

    # Resolve input files
    rush_paths = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.MOV", "*.MP4"):
                rush_paths.extend(str(f) for f in p.glob(ext))
        elif p.is_file():
            rush_paths.append(str(p))
        else:
            print(f"Warning: {inp} is not a file or directory, skipping")

    if not rush_paths:
        print("Error: No valid video files found.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Auto Video Montage — {len(rush_paths)} rush file(s)")
    print(f"{'='*60}\n")
    for p in rush_paths:
        print(f"  • {p}")
    print()

    from src.orchestrator import Orchestrator

    orchestrator = Orchestrator(max_iterations=args.max_iter)

    def progress(node: str, message: str):
        # Already printed by state.log(), nothing extra needed for CLI
        pass

    state = orchestrator.run(rush_paths, progress_callback=progress)

    print(f"\n{'='*60}")
    if state.final_output_path:
        print(f"  SUCCESS: {state.final_output_path}")
        if state.critic_feedback:
            print(f"  Final score: {state.critic_feedback.score:.2f}")
        print(f"  Iterations: {state.iteration + 1}")
    else:
        print(f"  FAILED: {state.error}")
    print(f"{'='*60}\n")

    if args.dump_state:
        dump_path = Path("output") / f"state_{state.run_id[:8]}.json"
        dump_path.parent.mkdir(exist_ok=True)
        with open(dump_path, "w") as f:
            json.dump(state.model_dump(), f, indent=2, default=str)
        print(f"State dumped to: {dump_path}")

    return 0 if state.final_output_path else 1


if __name__ == "__main__":
    sys.exit(main())
