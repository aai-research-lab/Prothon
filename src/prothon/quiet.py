"""Keeping a library's chatter out of a user's terminal.

MDTraj reads several formats through VMD's molfile plugins, which are compiled
and announce themselves on the way past::

    dcdplugin) detected standard 32-bit DCD file of native endianness
    dcdplugin) CHARMM format DCD file (also NAMD 2.1 and later)

That is written straight to file descriptor 1 from C, so
``contextlib.redirect_stdout`` does not see it -- Python's ``sys.stdout`` is a
wrapper the C code never consults. Capturing it means redirecting the
descriptor itself, which is what :func:`quiet_c_output` does.

Two lines per trajectory sounds harmless and is not. Prothon's own output is a
few lines, so on a study of a dozen ensembles the plugin outnumbers the result;
it lands in the middle of a JSON document when ``--json`` is used, making the
output unparseable; and it appears in the middle of a progress display. None of
it tells a user anything they can act on.

**What is not suppressed.** Only the descriptor, only while a file is being
read, and only when Prothon is not being run verbosely. Anything the library
raises still raises, anything it logs through Python still logs, and
``--verbose`` turns the redirect off entirely so a genuine diagnostic from the
reader is visible when it is being looked for.
"""

from __future__ import annotations

import contextlib
import os
import sys

__all__ = ["quiet_c_output", "set_quiet"]

#: Turned off by ``--verbose``, so a reader's diagnostics are visible when
#: somebody is looking for them.
_ENABLED = True


def set_quiet(enabled: bool) -> None:
    """Whether to suppress compiled libraries' output at all."""
    global _ENABLED
    _ENABLED = enabled


@contextlib.contextmanager
def quiet_c_output(stream: str = "stdout"):
    """Send one file descriptor to nowhere for the duration of a block.

    Parameters
    ----------
    stream
        ``stdout``, ``stderr``, or ``both``.

    Notes
    -----
    The descriptor is duplicated before being replaced and restored in a
    ``finally``, so an exception inside the block cannot leave a process
    writing to ``/dev/null`` -- which would be a far worse failure than the
    noise this exists to remove.

    Nothing is suppressed when the interpreter's streams have already been
    replaced by something without a real descriptor, as pytest's capture does:
    ``fileno()`` raises there, and the block runs unchanged rather than
    fighting the harness.
    """
    if not _ENABLED:
        yield
        return

    targets = {"stdout": [1], "stderr": [2], "both": [1, 2]}.get(stream)
    if targets is None:
        raise ValueError(f"stream must be stdout, stderr or both; got {stream!r}.")

    saved: dict[int, int] = {}
    devnull = None
    try:
        for fd in targets:
            # A captured stream may have no real descriptor behind it.
            try:
                (sys.stdout if fd == 1 else sys.stderr).fileno()
            except (AttributeError, OSError, ValueError):
                continue
            if devnull is None:
                devnull = os.open(os.devnull, os.O_WRONLY)
            saved[fd] = os.dup(fd)
            os.dup2(devnull, fd)
        yield
    finally:
        for fd, original in saved.items():
            try:
                os.dup2(original, fd)
            finally:
                os.close(original)
        if devnull is not None:
            os.close(devnull)
