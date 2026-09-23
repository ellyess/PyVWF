"""Run a command that writes manifests, holding the run lock while it runs.

A run stamps `git_dirty` into every manifest it writes, from
`git status --porcelain`, so a file created or edited in the tree mid-run marks
the whole run dirty. This wrapper refuses to start on a dirty tree, writes
`output/.run-lock` for the run's duration, and reports afterwards if the tree
changed during the run. The Claude Code hook `.claude/hooks/guard_run_lock.py`
blocks edits to the tree while the lock is held. `docs/design/agent-guards.md`
says why.

    python scripts/dev/run_locked.py -- pyvwf-validate train --region configs/regions/nz.toml
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

LOCK = Path("output/.run-lock")


def _porcelain(root: Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
    ).stdout


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def lock_holder(root: Path) -> dict | None:
    """The live lock's record, or None when there is no lock or it is stale.

    An unreadable lock counts as held, so a damaged file fails closed.
    """
    path = root / LOCK
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        pid = int(record["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return {"pid": None, "command": "unreadable lock file"}
    return record if pid_alive(pid) else None


def main(command: list[str], *, allow_dirty: bool = False) -> int:
    """Run ``command`` from the repository root holding the lock; return its exit code."""

    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    if Path.cwd().resolve() != root.resolve():
        print(f"Start runs from the repository root: {root}", file=sys.stderr)
        return 2
    holder = lock_holder(root)
    if holder is not None:
        print(f"A run already holds the lock: {holder}", file=sys.stderr)
        return 2
    if _porcelain(root) and not allow_dirty:
        print("The tree is dirty; every manifest would say git_dirty: true.", file=sys.stderr)
        print(_porcelain(root), file=sys.stderr)
        return 2

    lock = root / LOCK
    lock.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "pid": os.getpid(),
        "command": " ".join(command),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    lock.write_text(json.dumps(record) + "\n", encoding="utf-8")
    try:
        code = subprocess.run(command, cwd=root).returncode
    finally:
        try:
            if json.loads(lock.read_text(encoding="utf-8")).get("pid") == os.getpid():
                lock.unlink()
        except (OSError, ValueError):
            pass

    changed = _porcelain(root)
    if changed and not allow_dirty:
        print(
            "\nThe tree changed during the run. Its manifests may say git_dirty: true:",
            file=sys.stderr,
        )
        print(changed, file=sys.stderr)
    return code


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", nargs="+", help="the run command, after --")
    parser.add_argument("--allow-dirty", action="store_true", help="start on a dirty tree")
    args = parser.parse_args(argv)
    return main(args.command, allow_dirty=args.allow_dirty)


if __name__ == "__main__":
    raise SystemExit(cli())
