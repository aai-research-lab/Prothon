"""Where a parameter is defined, once, for every interface that offers it.

  - :mod:`schema` -- the parameters and the subcommands

The command-line parser is generated from :data:`schema.PARAMETERS`, so a flag
cannot exist without a keyword argument of the same name.
"""

from .schema import COMMANDS, PARAMETERS, Command, Parameter, parameters_for

__all__ = ["COMMANDS", "PARAMETERS", "Command", "Parameter", "parameters_for"]
