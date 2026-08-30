"""One definition per parameter, and every interface reads it.

A command-line flag, a keyword argument and a form field are three renderings
of the same idea, and when each is written by hand they drift. Prothon's did:
the command line called it ``--seed`` and the API called it ``random_state``,
the command line said ``-traj`` and the constructor said ``traj_files``, and a
PED accession could be loaded from Python but not from a terminal.

So the parameters are declared once, here, and the parser is generated from
them. A flag cannot exist without a keyword argument of the same name, because
both are read from the same row.

The names are the Python ones. ``--random-state`` on the command line is
``random_state`` in Python; the hyphen is a convention of one interface rather
than a different parameter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = ["COMMANDS", "PARAMETERS", "Command", "Parameter", "parameters_for"]


@dataclass(frozen=True)
class Parameter:
    """One parameter, in every interface that offers it.

    Attributes
    ----------
    name
        The Python name. The command-line flag is this with underscores
        replaced by hyphens, so the two cannot disagree.
    short
        A one-letter flag, where one is worth having. Reserved letters are
        avoided: ``-h`` is help.
    help
        One line, shown by ``--help`` and used as the field label in a form.
    metavar
        What the value is called in usage text.
    """

    name: str
    help: str
    default: Any = None
    kind: type = str
    short: str | None = None
    choices: Sequence[str] | None = None
    metavar: str | None = None
    action: str | None = None
    nargs: str | None = None
    commands: tuple[str, ...] = ()

    @property
    def flag(self) -> str:
        return "--" + self.name.replace("_", "-")

    @property
    def flags(self) -> list[str]:
        return [self.flag] if self.short is None else [self.short, self.flag]


def _p(*args, **kwargs) -> Parameter:
    return Parameter(*args, **kwargs)


#: Every parameter Prothon offers, in the order a user meets them.
PARAMETERS: tuple[Parameter, ...] = (
    # -- what to compare --------------------------------------------------
    _p(
        "config", short="-c", metavar="PATH",
        help="A study in a YAML file: which ensembles, which order parameters, "
             "which settings. Anything also given as a flag overrides it.",
        commands=("compare",),
    ),
    _p(
        "ensembles", short="-e", nargs="+", metavar="SOURCE",
        help="Ensembles to compare, one per source. A trajectory file, a "
             "directory of single-model PDBs, a glob, a multi-model PDB, or a "
             "PED accession such as PED00024 or PED00001:e002.",
        commands=("compare", "validate"),
    ),
    _p(
        "topology", short="-t", nargs="+", metavar="PATH",
        help="Topology for sources that need one: a single path shared by "
             "every ensemble, or one per ensemble in the same order. Not "
             "required for PED accessions or multi-model PDBs, which carry "
             "their own.",
        commands=("compare", "validate"),
    ),
    _p(
        "chains", nargs="+", metavar="ID",
        help="Keep only these chains: a PDB chain letter or an index. One "
             "selection shared by every ensemble, or one per ensemble in the "
             "same order.",
        commands=("compare", "validate"),
    ),
    _p(
        "reference", short="-r", default="0", metavar="SOURCE",
        help="The ensemble the others are measured against: an index into "
             "--ensembles, or a source of its own.",
        commands=("compare",),
    ),
    # -- what to measure --------------------------------------------------
    _p(
        "order_parameters", short="-p", default="cbcn", metavar="NAME[,NAME...]",
        help="Local order parameters: cbcn, cacn, caba, cata, sasa. Several "
             "are cheap and usually informative.",
        commands=("compare",),
    ),
    _p(
        "metric", short="-m", default="jsd", choices=("jsd", "wasserstein", "ks"),
        help="Per-residue distance. 'wasserstein' reports in the feature's "
             "own units; 'jsd' and 'ks' are bounded in [0, 1].",
        commands=("compare",),
    ),
    # -- how carefully ----------------------------------------------------
    _p(
        "random_state", short="-s", kind=int, metavar="N",
        help="Random seed. Set it and the run is reproducible.",
        commands=("compare", "validate"),
    ),
    _p(
        "n_permutations", kind=int, default=100, metavar="N",
        help="Relabellings behind the null. 100 rejects about 6%% where 5%% is "
             "asked for; raise it for a published result.",
        commands=("compare",),
    ),
    _p(
        "s_num", kind=int, default=5, metavar="N",
        help="Split-half repeats behind the noise floor.",
        commands=("compare",),
    ),
    _p(
        "x_num", kind=int, default=100, metavar="N",
        help="Grid points per estimated density.",
        commands=("compare",),
    ),
    _p(
        "alpha", kind=float, default=0.05, metavar="P",
        help="False-discovery rate for the per-residue test.",
        commands=("compare",),
    ),
    _p(
        "block_permutation", action="store_true",
        help="Force block permutation on. By default it is used whenever a "
             "correlation time longer than one frame is detected.",
        commands=("compare",),
    ),
    _p(
        "no_block_permutation", action="store_true",
        help="Treat frames as independent. Right for generated structures or "
             "an already-subsampled trajectory; wrong for a trajectory.",
        commands=("compare",),
    ),
    _p(
        "legacy_statistics", action="store_true",
        help="Reproduce the historical statistics, for regenerating a "
             "published figure. Documented as unsound.",
        commands=("compare",),
    ),
    # -- what to produce --------------------------------------------------
    _p(
        "save_config", metavar="PATH",
        help="Write the study this command describes to a file, so a command "
             "line typed once can be committed beside the manuscript.",
        commands=("compare",),
    ),
    _p(
        "report", default="summary", choices=("summary", "table"),
        help="How to present the results. 'table' ranks the ensembles by the "
             "margin above each one's own noise floor and adds coverage and "
             "fidelity: the view for several ensembles against a reference.",
        commands=("compare",),
    ),
    _p(
        "output_dir", short="-o", metavar="DIR",
        help="Where to write results. Defaults to the working directory.",
        commands=("compare", "validate"),
    ),
    _p(
        "dimred", short="-d", default="none", metavar="NAME[,NAME...]",
        help="Projections to draw: pca, mds, tsne, or none.",
        commands=("compare",),
    ),
    _p(
        "json", action="store_true",
        help="Print results as JSON instead of a summary.",
        commands=("compare", "validate"),
    ),
    _p(
        "verbose", short="-v", action="store_true",
        help="Verbose logging.",
        commands=("compare", "benchmark", "validate", "info"),
    ),
    # -- validate ---------------------------------------------------------
    _p(
        "observable", default="rg",
        choices=("rg", "end_to_end", "j_hn_ha"),
        help="What to compute and score against measurements.",
        commands=("validate",),
    ),
    _p(
        "experimental", metavar="PATH",
        help="Measured values, one per line, or two columns of value and "
             "uncertainty.",
        commands=("validate",),
    ),
    _p(
        "uncertainty", metavar="PATH_OR_VALUE",
        help="Experimental uncertainties: a file, or one number applied to "
             "every measurement.",
        commands=("validate",),
    ),
)


@dataclass(frozen=True)
class Command:
    """One subcommand."""

    name: str
    help: str
    description: str = ""
    runner: Callable | None = field(default=None, compare=False)


#: The subcommands, each drawing its flags from :data:`PARAMETERS`.
COMMANDS: tuple[Command, ...] = (
    Command(
        "compare",
        "Compare two or more ensembles.",
        "Every source is one ensemble; they are never concatenated. Each "
        "result is reported beside the smallest difference the sampling can "
        "resolve.\n\n"
        "With --reference naming a source and --report table, this is a "
        "benchmark: several ensembles against one, ranked by the margin above "
        "each one's own floor. There is no separate command for it, because a "
        "benchmark is this comparison presented differently.",
    ),
    Command(
        "validate",
        "Score an ensemble against experimental measurements.",
        "Reported beside a floor obtained from the ensemble itself, because a "
        "perfect ensemble does not score a reduced chi-squared of one.",
    ),
    Command(
        "info",
        "Show the order parameters, metrics, sources and detected backends.",
    ),
)


def parameters_for(command: str) -> tuple[Parameter, ...]:
    """The parameters one subcommand offers."""
    return tuple(p for p in PARAMETERS if command in p.commands)
