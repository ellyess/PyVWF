"""Run CI's lint, type, test and docs checks locally, in CI's pinned environment.

The project environment is not CI's: it runs a newer Python and pandas 2,
and it has the `data` extra installed, which CI's test and lint jobs do not.
On 2026-10-16 two realdata stamps were spent on commits CI then failed, on an
empty datetime map under pandas 3 and on mypy without `entsoe` installed; a
Python 3.12 environment built from `constraints/py312.txt` showed both in
under a minute. Run this before `scripts/dev/stamp.py realdata`.

    python scripts/dev/ci_local.py
    python scripts/dev/ci_local.py --skip docs --skip pytest

It builds two uv environments for Python 3.12 under `output/ci-local/`, which
is git-ignored, and reuses them on later runs: `.[dev]` with
`constraints/py312.txt`, as the lint job and the 3.12 test cell install it,
and `.[docs]` with `constraints/py312-docs.txt`, as the docs job does. Each
run reinstalls into them, which is quick when nothing changed. Each check
prints one PASS or FAIL line and writes its whole output to
`output/ci-local/logs/<check>.log`; a failure also prints the end of it. The
exit code is 1 if any check failed. Needs `uv`.

The constraints are resolved for Linux, so a package that only ships Linux
wheels at its pinned version would fail to install here; none does today.
`PYVWF_INPUT` is removed from the checks' environment, since CI has none.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "output" / "ci-local"
PYTHON = "3.12"


class Env(NamedTuple):
    extras: str
    constraints: str


#: Constraint paths are relative to ROOT, where uv runs: the repository path
#: may hold spaces, which uv splits a constraint path on.
ENVS = {
    "dev": Env(".[dev]", "constraints/py312.txt"),
    "docs": Env(".[docs]", "constraints/py312-docs.txt"),
}

#: name -> (environment, command), in the order CI's jobs run them. The
#: commands are the workflow's own; `tests/test_ci_local.py` checks that each
#: still appears in `.github/workflows/ci.yml`.
CHECKS = {
    "ruff check": ("dev", ["ruff", "check", "src", "tests", "scripts", "examples"]),
    "ruff format": ("dev", ["ruff", "format", "--check", "src", "tests", "scripts", "examples"]),
    "mypy": ("dev", ["mypy"]),
    "lint-imports": ("dev", ["lint-imports"]),
    "deptry": ("dev", ["deptry", "src"]),
    "vulture": ("dev", ["vulture"]),
    "pytest": (
        "dev",
        ["pytest", "tests", "-n", "4", "--dist", "loadfile", "-m", "not realdata"],
    ),
    "docs": ("docs", ["sphinx-build", "-b", "html", "docs", "{out}", "-W"]),
}

#: Lines of a failing check's output printed to the terminal.
TAIL = 40


def _venv(name: str) -> Path:
    return BASE / f"venv-py{PYTHON.replace('.', '')}-{name}"


def _prepare(name: str, recreate: bool) -> tuple[bool, str]:
    """Create or reuse one environment and install into it; (ok, output)."""
    env, venv = ENVS[name], _venv(name)
    if recreate and venv.exists():
        shutil.rmtree(venv)
    steps = []
    if not (venv / "bin" / "python").exists():
        steps.append(["uv", "venv", "--python", PYTHON, str(venv)])
    steps.append(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv / "bin" / "python"),
            "-e",
            env.extras,
            "-c",
            env.constraints,
        ]
    )
    output = []
    for step in steps:
        proc = subprocess.run(step, cwd=ROOT, capture_output=True, text=True)
        output.append(f"$ {' '.join(step)}\n{proc.stdout}{proc.stderr}")
        if proc.returncode != 0:
            return False, "".join(output)
    return True, "".join(output)


def _environ(venv: Path) -> dict[str, str]:
    environ = {k: v for k, v in os.environ.items() if k not in ("PYVWF_INPUT", "PYTHONPATH")}
    environ["VIRTUAL_ENV"] = str(venv)
    environ["PATH"] = f"{venv / 'bin'}{os.pathsep}{environ.get('PATH', '')}"
    environ["MPLBACKEND"] = "Agg"
    return environ


def _report(name: str, ok: bool, seconds: float, output: str, log: Path) -> None:
    log.write_text(output, encoding="utf-8")
    print(f"{'PASS' if ok else 'FAIL'}  {name:<14} {seconds:6.1f} s  {log.relative_to(ROOT)}")
    if not ok:
        for line in output.splitlines()[-TAIL:]:
            print(f"      | {line}")
    sys.stdout.flush()


def main(skip: set[str], recreate: bool) -> int:
    if shutil.which("uv") is None:
        print("ci_local needs uv on PATH: https://docs.astral.sh/uv/", file=sys.stderr)
        return 2
    logs = BASE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for old in logs.glob("*.log"):
        old.unlink()

    checks = {name: spec for name, spec in CHECKS.items() if name not in skip}
    results: dict[str, bool] = {}
    ready: dict[str, bool] = {}
    for name in sorted({env for env, _ in checks.values()}, key=list(ENVS).index):
        started = time.monotonic()
        ok, output = _prepare(name, recreate)
        _report(
            f"install {name}", ok, time.monotonic() - started, output, logs / f"install-{name}.log"
        )
        ready[name] = results[f"install {name}"] = ok

    if ready.get("dev"):
        probe = subprocess.run(
            [
                str(_venv("dev") / "bin" / "python"),
                "-c",
                "import sys, pandas, numpy; print('python', sys.version.split()[0], "
                "'pandas', pandas.__version__, 'numpy', numpy.__version__)",
            ],
            capture_output=True,
            text=True,
        )
        print(f"      {probe.stdout.strip() or probe.stderr.strip()}")

    with tempfile.TemporaryDirectory(prefix="pyvwf-docs-") as out:
        for name, (env, command) in checks.items():
            if not ready.get(env):
                print(f"SKIP  {name:<14} the {env} environment did not install")
                results[name] = False
                continue
            command = [arg.replace("{out}", out) for arg in command]
            started = time.monotonic()
            proc = subprocess.run(
                command,
                cwd=ROOT,
                env=_environ(_venv(env)),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            output = f"$ {' '.join(command)}\n{proc.stdout}"
            slug = name.replace(" ", "-")
            _report(
                name, proc.returncode == 0, time.monotonic() - started, output, logs / f"{slug}.log"
            )
            results[name] = proc.returncode == 0

    failed = [name for name, ok in results.items() if not ok]
    if failed:
        print(f"\n{len(failed)} of {len(results)} failed: {', '.join(failed)}")
        return 1
    print(f"\nAll {len(results)} passed.")
    return 0


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--skip", action="append", default=[], choices=sorted(CHECKS), help="a check to leave out"
    )
    parser.add_argument(
        "--recreate", action="store_true", help="delete and rebuild the environments first"
    )
    args = parser.parse_args(argv)
    return main(set(args.skip), args.recreate)


if __name__ == "__main__":
    raise SystemExit(cli())
