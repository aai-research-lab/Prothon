"""Where a parameter is defined, once, for every interface that offers it.

  - :mod:`schema` -- the parameters and the subcommands
  - :mod:`study`  -- a whole comparison, read from a file

The command-line parser is generated from :data:`schema.PARAMETERS`, so a flag
cannot exist without a keyword argument of the same name.
"""

from .schema import COMMANDS, PARAMETERS, Command, Parameter, parameters_for
from .study import Study, load_study, resolve_ensembles

__all__ = [
    "COMMANDS",
    "PARAMETERS",
    "Command",
    "Parameter",
    "Study",
    "load_study",
    "parameters_for",
    "resolve_ensembles",
]
