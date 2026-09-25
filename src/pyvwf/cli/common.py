"""Shared pieces for PyVWF's command-line entry points.

Every entry point, in ``src/pyvwf/cli/`` and in ``scripts/``, is meant to have
the same shape: a module docstring that says what it does and when to run it,
``argparse`` for every input, a ``main()`` and a ``__main__`` guard, and no
domain logic of its own. These helpers keep the repeated parts in one place.

The one that changes behaviour is :func:`input_path`. A default that names
``input/...`` literally ignores ``PYVWF_INPUT``, so with
``PYVWF_INPUT=input/combined`` a fetch script would write to one tree and the
matching process script read the other. Defaults built with
:func:`add_input_path` resolve under the input root the loaders use.

:func:`run` announces that root before the entry point does anything, because
which root a run read is invisible in its output and decides its numbers. The
curve library differs between ``input/`` and ``input/combined``, so the same
row rerun under the wrong one answers differently, and the difference reads
exactly like a code change. That has happened: a UK pin was reported as moving
by 0.0014 in capacity-factor MBE when the row had been rerun under the default
root instead of ``input/combined``, and the real movement was 1e-16. One line
of provenance on every run makes it visible at the top of the log.

Registered study constants are not arguments. A pre-registration fixes its
seeds, gates and grids in code; a flag would let a run differ from its record
without trace (``scripts/studies/README.md``).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

from pyvwf.config import PyVWFPaths


def make_parser(doc: str | None, *, prog: str | None = None) -> argparse.ArgumentParser:
    """A parser whose help is the entry point's module docstring, verbatim.

    Announces the input root as a side effect. Not every entry point goes
    through :func:`run` (several parse and dispatch themselves), but every one
    builds its parser here, so this is the hook that reaches all of them.
    """
    announce_input_root()
    return argparse.ArgumentParser(
        prog=prog,
        description=doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def resolved_input_root() -> tuple[Path, str]:
    """The input root this run reads, and how it was chosen.

    Returns:
        ``(root, provenance)``. Provenance is ``"PYVWF_INPUT"`` where the
        environment names the root, and ``"default"`` where nothing does and
        it falls back to ``input/``.
    """
    root = Path(PyVWFPaths.INPUT_ROOT)
    return root, "PYVWF_INPUT" if os.environ.get("PYVWF_INPUT") else "default"


#: Attribute holding the flags :func:`add_input_path` put on a parser. Only
#: those default under the input root, so only those can contradict it; an
#: output path given with a flag is not the root's business.
_INPUT_FLAGS = "_pyvwf_input_flags"


def _overriding_path_flags(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[str]:
    """Input-path flags given on the command line that sit outside the root.

    A flag from :func:`add_input_path` defaults under the input root, so a run
    that overrides one reads data the root does not describe. A path still
    inside the root is what the root already said, and is not reported.
    """
    flags = getattr(parser, _INPUT_FLAGS, None)
    if not flags:
        return []
    # abspath, not resolve: an input root may be assembled from symbolic links
    # to the shared era5/ and observations/ trees, which the training guide
    # recommends, and following them would call every such path an override.
    root = Path(os.path.abspath(PyVWFPaths.INPUT_ROOT))
    out = []
    for dest, option, default in flags:
        value = getattr(args, dest, None)
        if not isinstance(value, Path) or value == default:
            continue
        try:
            Path(os.path.abspath(value)).relative_to(root)
        except ValueError:
            out.append(f"{option}={value}")
    return out


#: Set once the root has been announced, so an entry point that both builds a
#: parser here and dispatches through :func:`run` says it once.
_announced = False


def announce_input_root(*, stream=None, force: bool = False) -> None:
    """Write one line naming the input root and how it was chosen.

    To stderr, not stdout: some entry points print a path or a table on stdout
    that a caller parses (``tests/test_pin_offset_fits.py`` reads the last
    line), and a banner must not enter that. Set ``PYVWF_QUIET_ROOT`` to skip
    it where a caller wants a silent run.

    Says it once per process unless ``force``.
    """
    global _announced
    if os.environ.get("PYVWF_QUIET_ROOT") or (_announced and not force):
        return
    _announced = True
    root, how = resolved_input_root()
    print(f"input root: {root} ({how})", file=stream if stream is not None else sys.stderr)


def announce_path_overrides(
    parser: argparse.ArgumentParser, args: argparse.Namespace, *, stream=None
) -> None:
    """Name any path argument given on the command line that leaves the root.

    Silent when there is none, which is the usual case.
    """
    if os.environ.get("PYVWF_QUIET_ROOT"):
        return
    overrides = _overriding_path_flags(parser, args)
    if overrides:
        print(
            "outside the input root: " + ", ".join(overrides),
            file=stream if stream is not None else sys.stderr,
        )


def input_path(*parts: str) -> Path:
    """A path under the input root (``PYVWF_INPUT``, default ``input/``).

    Read when called, so it follows the environment of the run rather than of
    the import.
    """
    return Path(PyVWFPaths.INPUT_ROOT).joinpath(*parts)


def add_input_path(
    parser: argparse.ArgumentParser, flag: str, *parts: str, help: str | None = None
) -> None:
    """Add ``flag`` with a default of :func:`input_path` of ``parts``."""
    default = input_path(*parts)
    text = f"{help} " if help else ""
    action = parser.add_argument(
        flag, type=Path, default=default, help=f"{text}(default: {default})"
    )
    # Recorded so a run that sends one of these outside the root says so: the
    # announced root would otherwise describe data the run did not read.
    recorded = getattr(parser, _INPUT_FLAGS, None)
    if recorded is None:
        recorded = []
        setattr(parser, _INPUT_FLAGS, recorded)
    recorded.append((action.dest, flag, default))


def add_region(
    parser: argparse.ArgumentParser, *, flag: str = "--region", required: bool = True
) -> None:
    """Add the region config argument: a TOML file under ``configs/regions/``."""
    parser.add_argument(
        flag, type=Path, required=required, help="Region config TOML, e.g. configs/regions/nz.toml"
    )


def run(
    main: Callable[[argparse.Namespace], int | None],
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
) -> int:
    """Parse ``argv``, call ``main`` with the arguments, return its exit code.

    ``main`` returning None counts as success.
    """
    args = parser.parse_args(argv)
    announce_input_root()
    announce_path_overrides(parser, args)
    code = main(args)
    return 0 if code is None else int(code)


def entry(
    main: Callable[[argparse.Namespace], int | None], parser: argparse.ArgumentParser
) -> None:
    """Run :func:`run` on the process's arguments and exit with its code."""
    sys.exit(run(main, parser))
