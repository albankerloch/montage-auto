"""CLI entry point for the video montage pipeline."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.assemble import DEFAULT_BEAM, PRESETS
from src.config import TARGET_MONTAGE_DURATION

load_dotenv()


def _run_graph(args, rush_paths) -> int:
    """Moteur graphe : aucune boucle, sortie classée."""
    from src.pipeline import Run, run as run_pipeline

    preset_names = [p.strip() for p in args.presets.split(",") if p.strip()]
    unknown = [p for p in preset_names if p not in PRESETS]
    if unknown:
        print(f"Error: preset(s) inconnu(s): {', '.join(unknown)}")
        print(f"Disponibles: {', '.join(PRESETS)}")
        return 1

    if args.explain:
        r = Run(rush_paths, preset_names=preset_names, target_duration=args.duration, verbose=False)
        print(r.plan("ranked"))
        todo = r.todo("ranked")
        print(f"\n{len(todo)} nœud(s) à recalculer :")
        for n in todo:
            print(f"  • {n.name}:{n.key()[:8]}" + (f" [{n.label}]" if n.label else ""))
        print("\n(✓ = déjà en cache, • = à calculer)")
        return 0

    r = run_pipeline(
        rush_paths,
        preset_names,
        render=not args.no_render,
        export=True,
        target_duration=args.duration,
    )

    if args.resolve:
        from src.export_resolve import build_timeline_in_resolve
        from src.models import EditPlan

        plan = EditPlan.model_validate(r.get("ranked")["plans"][0])
        print("\nBuilding timeline in DaVinci Resolve…")
        try:
            build_timeline_in_resolve(plan)
        except SystemExit as e:
            print(f"Resolve build skipped: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"Resolve build failed: {e}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Automatic video montage using AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main rushes/clip1.mp4 rushes/clip2.mp4
  python -m src.main rushes/*.mp4 --max-iter 2
  python -m src.main rushes/ --output output/my_montage.mp4

  # moteur graphe + solveur + faisceau (défaut)
  python -m src.main rushes/ --engine graph
  python -m src.main rushes/ --engine graph --presets punchy,emotional_arc --duration 90
  python -m src.main rushes/ --engine graph --explain      # que recalculerait-on ?

  # ancienne machine à états avec boucle de révision
  python -m src.main rushes/ --engine loop --max-iter 3
        """,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Video files or a directory containing video files",
    )
    parser.add_argument(
        "--engine",
        choices=("graph", "loop"),
        default="graph",
        help="graph = graphe de dépendances + CP-SAT + faisceau classé (défaut) ; "
             "loop = machine à états historique avec boucle CRITIC→REVISION",
    )
    parser.add_argument(
        "--presets",
        type=str,
        default=",".join(DEFAULT_BEAM),
        help=f"Intentions du faisceau, séparées par des virgules. Disponibles : "
             f"{', '.join(PRESETS)}",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=TARGET_MONTAGE_DURATION,
        help="Durée cible du montage en secondes (moteur graph)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Produire plan et exports NLE sans rendre le mp4 (moteur graph)",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Afficher le graphe et les nœuds à recalculer, puis sortir",
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
    parser.add_argument(
        "--resolve",
        action="store_true",
        help="After a successful run, build the timeline directly inside "
             "DaVinci Resolve via the scripting API (Resolve must be open, "
             "external scripting set to Local)",
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

    if args.engine == "graph":
        return _run_graph(args, rush_paths)

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

    if args.resolve and state.final_output_path and state.edit_plan:
        print("Building timeline in DaVinci Resolve…")
        try:
            from src.export_resolve import build_timeline_in_resolve
            build_timeline_in_resolve(state.edit_plan)
        except SystemExit as e:
            print(f"Resolve build skipped: {e}")
        except Exception as e:
            print(f"Resolve build failed: {e}")

    if args.dump_state:
        dump_path = Path("output") / f"state_{state.run_id[:8]}.json"
        dump_path.parent.mkdir(exist_ok=True)
        with open(dump_path, "w") as f:
            json.dump(state.model_dump(), f, indent=2, default=str)
        print(f"State dumped to: {dump_path}")

    return 0 if state.final_output_path else 1


if __name__ == "__main__":
    sys.exit(main())
