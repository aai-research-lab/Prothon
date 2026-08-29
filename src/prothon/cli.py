"""The ``prothon`` command.

    prothon -traj wt.dcd,mutant.dcd -top top.pdb -m cbcn

The flags of version 2.0 all still work. Two defaults changed, both because
the old ones were traps:

**Dimensionality reduction is off unless asked for.** It used to default to
``pca,mds,tsne``. MDS builds a dense frame-by-frame distance matrix, so the
default turned a two-minute comparison over a real trajectory into an
out-of-memory failure, and the projection is a picture rather than part of the
measurement.

**Results print as a readable summary**, with the full JSON written to the
manifest in the output directory. ``--json`` restores the old behaviour of
dumping everything to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import __version__
from .core.metrics import METRICS, describe_metric
from .core.prothon_core import DIMRED_TECHNIQUES, Prothon
from .core.representation import MEASURES, describe_measure
from .utils import configure_logging

__all__ = ["build_parser", "main"]


def _to_serialisable(value):
    """Recursively convert results, including NumPy arrays, for JSON."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _to_serialisable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serialisable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prothon",
        description="Compare protein conformational ensembles using local order parameters.",
        epilog="Measures:\n  " + "\n  ".join(describe_measure(m) for m in sorted(MEASURES)),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"prothon {__version__}")
    parser.add_argument(
        "--info", action="store_true",
        help="Print the available measures and installed backends, then exit.",
    )
    parser.add_argument(
        "--benchmark", metavar="DIR", nargs="+",
        help="Compare each of these against the reference on equal terms. Each "
             "is a trajectory file, or a directory of single-model PDBs as a "
             "generative model emits them. Requires --reference.",
    )
    parser.add_argument(
        "--reference", metavar="PATH",
        help="The ensemble the others are measured against, for --benchmark.",
    )
    parser.add_argument(
        "-traj", "--trajectories",
        help="Comma-separated trajectory files, one per ensemble.",
    )
    parser.add_argument("-top", "--topology", help="Topology file (PDB).")
    parser.add_argument(
        "-m", "--methods", default="cbcn",
        help="Comma-separated measures (default: cbcn).",
    )
    parser.add_argument(
        "-r", "--ref", type=int, default=0,
        help="Reference ensemble index (default: 0).",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Root output directory (default: <measure>_output in the working directory).",
    )
    parser.add_argument(
        "-d", "--dimred", default="none",
        help=f"Comma-separated projections ({', '.join(DIMRED_TECHNIQUES)}), "
             f"or 'none' (default). MDS is refused above 5000 frames.",
    )
    parser.add_argument(
        "--metric", default="jsd", choices=sorted(METRICS),
        help="Per-feature distance (default: jsd). 'wasserstein' reports in the "
             "feature's own units; 'jsd' and 'ks' are bounded in [0, 1].",
    )
    parser.add_argument(
        "--x-num", type=int, default=100,
        help="Grid points per estimated density (default: 100).",
    )
    parser.add_argument(
        "--s-num", type=int, default=5,
        help="Resamples per ensemble for the noise floor (default: 5).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="False-discovery rate for the per-residue test (default: 0.05).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed. Set it for a reproducible run.",
    )
    parser.add_argument(
        "--legacy-statistics", action="store_true",
        help="Reproduce version 2.0's statistics exactly (one pooled two-sided "
             "test, no per-residue correction, linear grid for torsions).",
    )
    parser.add_argument("--json", action="store_true", help="Print full results as JSON.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return parser


def _print_info() -> None:
    print(f"prothon {__version__}\n")
    print("Measures:")
    for name in sorted(MEASURES):
        print(f"  {describe_measure(name)}")
    print("\nMetrics:")
    for name in sorted(METRICS):
        print(f"  {describe_metric(name)}")
    print("\nBackends:")
    for module, purpose in (
        ("mdtraj", "trajectory I/O and geometry"),
        ("scipy", "density estimation and statistics"),
        ("sklearn", "dimensionality reduction"),
        ("matplotlib", "figures"),
    ):
        try:
            version = __import__(module).__version__
            print(f"  {module:<12} {version:<10} {purpose}")
        except Exception:
            print(f"  {module:<12} {'not installed':<10} {purpose}  -> pip install {module}")


def _load(path: str, topology: str | None):
    """A trajectory file, or a directory of single-model structures."""
    import os

    from .ingest import Ensemble

    label = os.path.basename(str(path).rstrip("/")) or str(path)
    if os.path.isdir(path) or any(c in str(path) for c in "*?"):
        return Ensemble.from_pdb_models(path, label=label)
    if topology is None:
        return Ensemble.from_pdb_models(path, label=label)
    return Ensemble.from_trajectory(path, topology, label=label)


def _run_benchmark(args) -> int:
    """Compare several ensembles against one reference."""
    from .batch import benchmark

    try:
        reference = _load(args.reference, args.topology)
        models = [_load(p, args.topology) for p in args.benchmark]
        result = benchmark(
            reference, models,
            measures=args.methods,
            random_state=args.seed,
            output_dir=args.output,
        )
    except (ValueError, FileNotFoundError) as error:
        print(f"prothon: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=float))
    else:
        print(result.summary())
        print()
        for measure in result.measures:
            print(result.table(measure))
            print()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.info:
        _print_info()
        return 0

    if args.benchmark:
        if not args.reference:
            parser.error("--benchmark requires --reference")
        return _run_benchmark(args)

    missing = [
        flag for flag, value in (("-traj", args.trajectories), ("-top", args.topology))
        if not value
    ]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")

    configure_logging(args.verbose)

    dimred = None if args.dimred.strip().lower() in {"none", ""} else args.dimred

    try:
        study = Prothon(
            traj_files=args.trajectories,
            topology=args.topology,
            output_dir=args.output,
            verbose=args.verbose,
            random_state=args.seed,
        )
        results = study.compare_ensembles(
            methods=args.methods,
            ref=args.ref,
            x_num=args.x_num,
            s_num=args.s_num,
            dimred=dimred,
            alpha=args.alpha,
            metric=args.metric,
            legacy=args.legacy_statistics,
        )
    except (ValueError, FileNotFoundError) as error:
        # These are the errors that mean the study was described wrongly. A
        # traceback would bury the one sentence that says what to change.
        print(f"prothon: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(_to_serialisable(results), indent=2))
    else:
        print(study.summary())

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
