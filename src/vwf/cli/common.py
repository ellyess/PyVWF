"""Shared pieces for PyVWF's command-line entry points.

Every entry point, in ``src/vwf/cli/`` and in ``scripts/``, is meant to have
the same shape: a module docstring that says what it does and when to run it,
``argparse`` for every input, a ``main()`` and a ``__main__`` guard, and no
domain logic of its own. These helpers keep the repeated parts in one place.

The one that changes behaviour is :func:`input_path`. A default that names
``input/...`` literally ignores ``PYVWF_INPUT``, so with
``PYVWF_INPUT=input/combined`` a fetch script would write to one tree and the
matching process script read the other. Defaults built with
:func:`add_input_path` resolve under the input root the loaders use.

Registered study constants are not arguments. A pre-registration fixes its
seeds, gates and grids in code; a flag would let a run differ from its record
without trace (``scripts/studies/README.md``).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from vwf.config import PyVWFPaths


def make_parser(doc: str | None, *, prog: str | None = None) -> argparse.ArgumentParser:
    """A parser whose help is the entry point's module docstring, verbatim."""
    return argparse.ArgumentParser(
        prog=prog,
        description=doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    parser.add_argument(flag, type=Path, default=default, help=f"{text}(default: {default})")


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
    code = main(args)
    return 0 if code is None else int(code)


def entry(
    main: Callable[[argparse.Namespace], int | None], parser: argparse.ArgumentParser
) -> None:
    """Run :func:`run` on the process's arguments and exit with its code."""
    sys.exit(run(main, parser))
