"""The ``prothon`` command.

    prothon compare  --ensembles wt.xtc mutant.xtc --topology top.pdb
    prothon compare  --config study.yml
    prothon compare  --ensembles bioemu/ alphaflow/ -r md.xtc --report table
    prothon validate --ensembles md.xtc -t top.pdb --experimental rg.txt
    prothon info

Every flag here is generated from :data:`prothon.config.schema.PARAMETERS`, so
the command line and the Python API cannot drift apart: ``--random-state`` on
the command line is ``random_state`` in Python because both read the same row.

The 2.x form -- ``prothon -traj a.dcd,b.dcd -top top.pdb`` with no subcommand --
still works and says once where it went.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections.abc import Sequence
from typing import Any

from . import __version__
from .compare.distance import METRICS, describe_metric
from .config.schema import COMMANDS, parameters_for
from .represent.order_parameters import ORDER_PARAMETERS, describe_order_parameter
from .utils import configure_logging, split_list_arg

__all__ = ["build_parser", "main"]


def _add(parser: argparse.ArgumentParser, command: str) -> None:
    """Give a parser the flags its subcommand declares."""
    for spec in parameters_for(command):
        kwargs: dict[str, Any] = {"help": spec.help, "dest": spec.name}
        # A configured study needs to distinguish an omitted option from an
        # explicitly typed value that happens to equal the schema default.
        # Attribute presence is that provenance. Other commands do not merge
        # with configuration and retain ordinary argparse defaults.
        if command == "compare":
            kwargs["default"] = argparse.SUPPRESS
        if spec.action:
            kwargs["action"] = spec.action
        else:
            kwargs["type"] = spec.kind
            if command != "compare":
                kwargs["default"] = spec.default
            if spec.nargs:
                kwargs["nargs"] = spec.nargs
            if spec.choices:
                kwargs["choices"] = list(spec.choices)
            if spec.metavar:
                kwargs["metavar"] = spec.metavar
        parser.add_argument(*spec.flags, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prothon",
        description="Compare protein conformational ensembles.",
        epilog=(
            "Order parameters:\n  "
            + "\n  ".join(describe_order_parameter(m) for m in sorted(ORDER_PARAMETERS))
            + "\n\nMetrics:\n  "
            + "\n  ".join(describe_metric(m) for m in sorted(METRICS))
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Without this, argparse resolves any unambiguous prefix -- and the
        # hidden 2.x flags below make `-t` ambiguous against `-traj` and
        # `-top`, so a subcommand's own short flag stops working.
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"prothon {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for command in COMMANDS:
        sub = subparsers.add_parser(
            command.name,
            help=command.help,
            description=command.description or command.help,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        _add(sub, command.name)

    return parser


def _legacy_parser() -> argparse.ArgumentParser:
    """The 2.x flags, in a parser of their own.

    Kept off the main parser because argparse resolves option prefixes across
    every parser in play, and `-traj`/`-top` make a subcommand's own `-t`
    ambiguous. A published command line keeps working; a new one is not
    shaped by it.
    """
    parser = argparse.ArgumentParser(prog="prothon", add_help=False)
    parser.add_argument("-traj", "--trajectories")
    parser.add_argument("-top", dest="legacy_topology")
    parser.add_argument("-m", "--methods")
    parser.add_argument("--measures", dest="measures_legacy")
    parser.add_argument("--seed", type=int)
    parser.add_argument("-o", "--output")
    parser.add_argument("--info", action="store_true")
    parser.add_argument("-r", "--ref", type=int)
    parser.add_argument("-d", "--dimred")
    parser.add_argument("--metric")
    parser.add_argument("--x-num", type=int, dest="x_num")
    parser.add_argument("--s-num", type=int, dest="s_num")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--legacy-statistics", action="store_true",
                        dest="legacy_statistics")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _serialisable(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {k: _serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialisable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _reference_index(reference, ensembles, topology, cache_dir=None):
    """A reference given as an index, or as a source of its own.

    Returns the ensembles with the reference first, and its index.
    """
    from .ingest.sources import resolve

    if reference is None:
        return ensembles, 0
    text = str(reference)
    if text.isdigit():
        index = int(text)
        if not 0 <= index < len(ensembles):
            raise ValueError(
                f"Reference index {index} is out of range for "
                f"{len(ensembles)} ensembles (0 to {len(ensembles) - 1})."
            )
        return ensembles, index
    # A source: prepend it, and it becomes the reference.
    return [resolve(text, topology, cache_dir=cache_dir), *ensembles], 0


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------
def run_compare(args) -> int:
    """Every path here builds a Study and runs that.

    Flags become a study; a file is read into one; a file plus flags is the
    file with the flags applied. There is one object to run, so the command
    line cannot come to offer a setting the file does not.
    """
    from .config.study import Study

    if getattr(args, "config", None):
        study = Study.from_file(args.config).merged_with(args)
    else:
        if not getattr(args, "ensembles", None):
            raise ValueError(
                "compare needs --ensembles, or --config naming a file that "
                "lists them."
            )
        study = Study.from_arguments(args)

    if getattr(args, "save_config", None):
        study.save(args.save_config)
        print(f"Wrote {args.save_config}", file=sys.stderr)

    if study.settings.get("report", getattr(args, "report", "summary")) == "table":
        # The same comparison, ranked, with coverage and fidelity beside each
        # row. Not a separate command: a benchmark is this view.
        return _run_table(args, study)

    comparison = study.run()
    print(
        json.dumps(_serialisable(comparison.comparison_results), indent=2)
        if getattr(args, "json", False)
        else comparison.summary()
    )
    return 0


def _run_table(args, study) -> int:
    """Every other ensemble against the reference, ranked.

    Ranked by the margin above each ensemble's own noise floor rather than by
    distance, because a thinly sampled ensemble carries a higher floor *and* a
    depressed distance, so a table of distances flatters it.
    """
    from .batch import benchmark

    ensembles = study.resolve()
    reference = study.reference_index()
    others = [e for i, e in enumerate(ensembles) if i != reference]
    if not others:
        raise ValueError("Nothing to compare against the reference.")

    result = benchmark(
        ensembles[reference], others, **study.benchmark_arguments()
    )
    if getattr(args, "json", False):
        print(json.dumps(result.to_dict(), indent=2, default=float))
    else:
        print(result.summary())
        for name in result.order_parameters:
            print()
            print(result.table(name))
    return 0


def _load_experimental_table(path, uncertainty=None):
    """Read measured values without losing their row/column orientation."""
    import numpy as np

    data = np.loadtxt(path, ndmin=2)
    if data.size == 0:
        raise ValueError(f"Experimental table {path!s} is empty.")
    if data.ndim != 2 or data.shape[1] not in {1, 2}:
        raise ValueError(
            f"Experimental table {path!s} has ambiguous shape {data.shape}. "
            "Use one value per row, or two columns containing value and "
            "uncertainty."
        )

    measured = np.asarray(data[:, 0], dtype=np.float64)
    if data.shape[1] == 2:
        if uncertainty is not None:
            raise ValueError(
                "Uncertainty was supplied both as the experimental table's "
                "second column and with --uncertainty. Use one source."
            )
        sigma = np.asarray(data[:, 1], dtype=np.float64)
    elif uncertainty is not None:
        try:
            sigma = np.full(measured.size, float(uncertainty))
        except (TypeError, ValueError):
            sigma = np.asarray(np.loadtxt(uncertainty), dtype=np.float64).ravel()
            if sigma.size == 1:
                sigma = np.full(measured.size, float(sigma[0]))
    else:
        raise ValueError(
            "No uncertainties. Give a second column in --experimental, or "
            "--uncertainty as a file or a single number. A chi-squared without "
            "them is a sum of squares in arbitrary units."
        )

    if not np.all(np.isfinite(measured)):
        raise ValueError("Experimental values must all be finite.")
    if sigma.size != measured.size:
        raise ValueError(
            f"{sigma.size} uncertainties for {measured.size} experimental "
            "values. Give one uncertainty, or one per value."
        )
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError(
            "Experimental uncertainties must all be finite and strictly positive."
        )
    return measured, sigma


def run_validate(args) -> int:
    from .ingest.sources import resolve_all
    from .validate import score_observable
    from .validate.observables import (
        end_to_end,
        j_coupling_hn_ha,
        radius_of_gyration,
    )

    if not args.experimental:
        raise ValueError("validate needs --experimental: the measured values.")

    ensembles = resolve_all(args.ensembles, args.topology, chains=args.chains)
    measured, sigma = _load_experimental_table(
        args.experimental, args.uncertainty
    )

    compute = {
        "rg": lambda t: radius_of_gyration(t)[:, None],
        "end_to_end": lambda t: end_to_end(t)[:, None],
    }

    results = []
    for ensemble in ensembles:
        feature_index = feature_labels = None
        if args.observable == "j_hn_ha":
            from .ingest import residue_identity

            predicted, residue_indices = j_coupling_hn_ha(ensemble.trajectory)
            feature_index, feature_labels = residue_identity(
                ensemble.topology, residue_indices
            )
        else:
            predicted = compute[args.observable](ensemble.trajectory)
        result = score_observable(
            predicted, measured, sigma,
            observable=f"{args.observable} [{ensemble.label}]",
            weights=ensemble.weights,
            labels=feature_labels,
            feature_index=feature_index,
            random_state=args.random_state,
        )
        results.append(result)
        if not args.json:
            print(result.summary())
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2, default=float))
    return 0


def run_info(args=None) -> int:
    print(f"prothon {__version__}\n")
    print("Commands:")
    for command in COMMANDS:
        print(f"  {command.name:<12} {command.help}")
    print("\nOrder parameters:")
    for name in sorted(ORDER_PARAMETERS):
        print(f"  {describe_order_parameter(name)}")
    print("\nMetrics:")
    for name in sorted(METRICS):
        print(f"  {describe_metric(name)}")
    print("\nSources accepted by --ensembles:")
    for line in (
        "a trajectory file, with --topology",
        "a directory of single-model PDBs",
        "a glob such as 'samples/*.pdb'",
        "a multi-model PDB",
        "a PED accession such as PED00024, or PED00001:e002",
    ):
        print(f"  {line}")
    print("\nBackends:")
    for module, purpose in (
        ("mdtraj", "trajectory I/O and geometry"),
        ("scipy", "density estimation and statistics"),
        ("sklearn", "dimensionality reduction and the classifier test"),
        ("matplotlib", "figures"),
    ):
        try:
            print(f"  {module:<12} {__import__(module).__version__:<10} {purpose}")
        except Exception:
            print(f"  {module:<12} {'not installed':<10} {purpose}")
    return 0


RUNNERS = {
    "compare": run_compare,
    "validate": run_validate,
    "info": run_info,
}


def _legacy(argv, parser) -> int:
    """The 2.x invocation, translated into the new one."""
    import warnings

    args, unknown = _legacy_parser().parse_known_args(argv)
    if args.info:
        return run_info()
    if not args.trajectories:
        parser.print_help()
        return 0
    if unknown:
        print(f"prothon: unrecognised arguments: {' '.join(unknown)}", file=sys.stderr)
        return 2

    warnings.warn(
        "`prothon -traj ... -top ...` is now `prothon compare --ensembles ... "
        "--topology ...`, and --seed is --random-state. The old form still "
        "works and will be removed in 4.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    namespace = argparse.Namespace(
        ensembles=split_list_arg(args.trajectories),
        topology=args.legacy_topology,
        reference=str(args.ref if args.ref is not None else 0),
        order_parameters=args.methods or getattr(args, "measures_legacy", None) or "cbcn",
        metric=args.metric or "jsd",
        random_state=args.seed,
        n_permutations=100,
        s_num=args.s_num if args.s_num is not None else 5,
        x_num=args.x_num if args.x_num is not None else 100,
        alpha=args.alpha if args.alpha is not None else 0.05,
        block_permutation=False,
        no_block_permutation=False,
        report="summary",
        legacy_statistics=args.legacy_statistics,
        output_dir=args.output,
        dimred=args.dimred or "none",
        json=args.json,
        verbose=args.verbose,
    )
    return run_compare(namespace)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    except BrokenPipeError:
        # `prothon info | head` closes the pipe while output is still being
        # written. Every Unix tool has to survive that quietly; a traceback
        # here would be the only thing a reader sees of an otherwise
        # successful run.
        #
        # The descriptor is redirected so the interpreter does not report the
        # same error again while flushing at shutdown. Under a test harness
        # stdout may have no real descriptor, in which case there is nothing
        # to redirect and nothing that will be flushed either.
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
            pass
        return 0


def _main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    commands = {c.name for c in COMMANDS}

    # No subcommand means either the 2.x form or nothing at all. Deciding this
    # before argparse runs keeps the old flags from shaping the new parser.
    if not any(token in commands for token in raw):
        if raw and raw[0] in {"--version"}:
            parser.parse_args(raw)          # exits
        try:
            return _legacy(raw, parser)
        except (ValueError, FileNotFoundError, TypeError) as error:
            print(f"prothon: {error}", file=sys.stderr)
            return 2

    args = parser.parse_args(raw)
    configure_logging(getattr(args, "verbose", False))

    try:
        return RUNNERS[args.command](args)
    except (ValueError, FileNotFoundError, TypeError) as error:
        # These mean the study was described wrongly. A traceback would bury
        # the one sentence that says what to change.
        print(f"prothon: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
