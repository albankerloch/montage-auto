"""CLI entry point for the video montage pipeline."""
from __future__ import annotations
import argparse
import os
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.assemble import DEFAULT_BEAM, PRESETS
from src.config import ANALYZER_MODEL, TARGET_MONTAGE_DURATION

load_dotenv()


VIDEO_EXTENSIONS = ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.MOV", "*.MP4")


def collect_rush_paths(inputs) -> list[str]:
    """Résout fichiers et dossiers en une liste de rushes.

    Extraite de main() pour que `src.bench_annot` s'en serve au lieu de
    reproduire la même boucle avec une liste d'extensions qui divergerait.
    """
    paths: list[str] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            for ext in VIDEO_EXTENSIONS:
                paths.extend(str(f) for f in p.glob(ext))
        elif p.is_file():
            paths.append(str(p))
        else:
            print(f"Warning: {inp} is not a file or directory, skipping")
    return paths


def _run_graph(args, rush_paths) -> int:
    """Moteur graphe : aucune boucle, sortie classée."""
    from src.pipeline import Run, run as run_pipeline

    if args.local_url:
        os.environ["LOCAL_VLM_BASE_URL"] = args.local_url

    annot_model = args.annot_model or ANALYZER_MODEL

    preset_names = [p.strip() for p in args.presets.split(",") if p.strip()]

    bans: list[str] = []
    for raw in args.ban:
        bans.extend(k.strip() for k in raw.split(",") if k.strip())
    if args.ban_file:
        ban_path = Path(args.ban_file)
        if not ban_path.exists():
            print(f"Error: {ban_path} introuvable")
            return 1
        bans.extend(json.loads(ban_path.read_text()))
    bans = sorted(set(bans))
    if bans:
        print(f"Veto sur {len(bans)} plan(s) : {', '.join(bans)}\n")

    pins: dict[str, int | str] = {}
    raw_pins: list[str] = []
    for raw in args.pin:
        raw_pins.extend(k.strip() for k in raw.split(",") if k.strip())
    if args.pin_file:
        pin_path = Path(args.pin_file)
        if not pin_path.exists():
            print(f"Error: {pin_path} introuvable")
            return 1
        for key, pos in json.loads(pin_path.read_text()).items():
            raw_pins.append(f"{key}={pos}")
    for entry in raw_pins:
        if "=" not in entry:
            print(f"Error: --pin attend CLE=POSITION (p. ex. rush_0@12.250=0) : {entry!r}")
            return 1
        key, _, pos_raw = entry.partition("=")
        key, pos_raw = key.strip(), pos_raw.strip()
        if pos_raw in ("first", "last"):
            pins[key] = pos_raw
        else:
            try:
                pins[key] = int(pos_raw)
            except ValueError:
                print(f"Error: position de pin invalide pour {key!r} : {pos_raw!r} "
                      "(entier 0-based, ou 'first'/'last')")
                return 1
    if pins:
        print(f"Pin sur {len(pins)} plan(s) : "
              + ", ".join(f"{k}=@{v}" for k, v in sorted(pins.items())) + "\n")

    unknown = [p for p in preset_names if p not in PRESETS]
    if unknown:
        print(f"Error: preset(s) inconnu(s): {', '.join(unknown)}")
        print(f"Disponibles: {', '.join(PRESETS)}")
        return 1

    if args.explain:
        r = Run(
            rush_paths,
            preset_names=preset_names,
            target_duration=args.duration,
            banned_segments=bans,
            pinned_segments=pins,
            rank_mode=args.rank,
            pick=args.pick,
            annot_model=annot_model,
            thumbnail_width=args.thumbnail_width,
            verbose=False,
        )
        print(r.plan("ranked"))
        todo = r.todo("ranked")
        print(f"\n{len(todo)} nœud(s) à recalculer :")
        for n in todo:
            print(f"  • {n.name}:{n.key()[:8]}" + (f" [{n.label}]" if n.label else ""))
        print("\n(✓ = déjà en cache, • = à calculer)")
        return 0

    try:
        r = run_pipeline(
            rush_paths,
            preset_names,
            render=not args.no_render,
            export=True,
            target_duration=args.duration,
            banned_segments=bans,
            pinned_segments=pins,
            rank_mode=args.rank,
            pick=args.pick,
            annot_model=annot_model,
            thumbnail_width=args.thumbnail_width,
        )
    except ValueError as e:
        print(f"\nError: {e}")
        return 1

    if args.resolve and args.rank == "manual":
        print("\n--resolve ignoré en classement manuel : choisir d'abord avec --pick.")
        return 0

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

  # boucle humaine : arbitrer soi-même, puis vetoer et rejouer
  python -m src.main rushes/ --rank manual                 # sort les K candidats en EDL
  python -m src.main rushes/ --pick 2 --ban rush_0@12.250,rush_1@3.000
  python -m src.main rushes/ --ban-file bans.json

  # annotation sur un VLM local au lieu d'Anthropic
  python -m src.main rushes/ --annot-model local/Qwen/Qwen3-VL-8B-Instruct
  python -m src.main rushes/ --annot-model local/qwen3-vl --local-url http://localhost:11434/v1

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
        "--annot-model",
        type=str,
        default=None,
        help="Modèle d'annotation. Préfixer par « local/ » pour viser un serveur "
             "compatible OpenAI (vLLM, Ollama, llama.cpp, LM Studio) au lieu "
             "d'Anthropic. Ex. : local/Qwen/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument(
        "--local-url",
        type=str,
        default=None,
        help="URL du serveur local (défaut : LOCAL_VLM_BASE_URL, "
             "http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--thumbnail-width",
        type=int,
        default=None,
        help="Largeur des vignettes envoyées à la vision. Défaut : 640 en API, "
             "1280 en local — un serveur local ne facture pas au token",
    )
    parser.add_argument(
        "--ban",
        action="append",
        default=[],
        metavar="CLE",
        help="Exclure un plan, par sa clé telle qu'affichée dans le rapport "
             "(p. ex. rush_0@12.250). Répétable, ou liste séparée par des virgules",
    )
    parser.add_argument(
        "--ban-file",
        type=str,
        default=None,
        help="Fichier JSON contenant une liste de clés à exclure (cumulé avec --ban)",
    )
    parser.add_argument(
        "--pin",
        action="append",
        default=[],
        metavar="CLE=POSITION",
        help="Imposer un plan à une position donnée (contrainte dure du solveur, "
             "pas une suggestion). POSITION est un entier 0-based, ou 'first'/'last' "
             "(p. ex. rush_0@12.250=0 ou rush_0@12.250=last). Répétable, ou liste "
             "séparée par des virgules",
    )
    parser.add_argument(
        "--pin-file",
        type=str,
        default=None,
        help="Fichier JSON {clé: position} (cumulé avec --pin)",
    )
    parser.add_argument(
        "--rank",
        choices=("llm", "manual"),
        default="llm",
        help="llm = classement par comparaison par paires ; manual = aucun appel "
             "modèle, les candidats sont exportés en EDL/FCPXML pour arbitrage",
    )
    parser.add_argument(
        "--pick",
        type=int,
        default=0,
        help="Index du candidat à rendre et exporter (défaut 0 = le premier)",
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
    rush_paths = collect_rush_paths(args.inputs)

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
