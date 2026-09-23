"""PreToolUse guard: no edits to the tree while a run holds the lock.

A run started through `scripts/dev/run_locked.py` holds `output/.run-lock`.
While it does, this hook blocks Write, Edit, MultiEdit and NotebookEdit on any
path inside the repository that git does not ignore, and blocks the git
commands that change the tree. Anything else in Bash is not caught; the rule
in AGENTS.md still covers it. `docs/design/agent-guards.md` says why.

Exit 2 blocks the tool call and shows stderr to Claude. Any unexpected error
exits 1, which lets the call through and shows the error to the user.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

GIT_WRITES = re.compile(
    r"\bgit\b[^;&|]*?\b(add|am|apply|checkout|cherry-pick|clean|commit|merge|mv|pull"
    r"|rebase|reset|restore|revert|rm|stash|switch)\b"
)


def _root(event: dict) -> Path:
    return Path(os.environ.get("CLAUDE_PROJECT_DIR") or event.get("cwd") or ".").resolve()


def _holder(root: Path) -> dict | None:
    sys.path.insert(0, str(root / "scripts" / "dev"))
    from run_locked import lock_holder

    return lock_holder(root)


def _ignored(root: Path, path: Path) -> bool:
    return subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=root).returncode == 0


def main() -> int:
    event = json.load(sys.stdin)
    root = _root(event)
    holder = _holder(root)
    if holder is None:
        return 0

    tool = event.get("tool_name", "")
    tool_input = event.get("tool_input") or {}
    running = holder.get("command", "a run")

    if tool == "Bash":
        command = tool_input.get("command", "")
        if GIT_WRITES.search(command):
            print(
                f"Blocked: `{running}` holds the run lock, and this git command changes "
                "the tree, which marks every manifest of the run git_dirty. Wait for the "
                "run to finish.",
                file=sys.stderr,
            )
            return 2
        return 0

    target = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not target:
        return 0
    path = Path(target)
    path = (path if path.is_absolute() else root / path).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return 0
    if _ignored(root, path):
        return 0
    print(
        f"Blocked: `{running}` holds the run lock. Writing {path.relative_to(root)} now "
        "would mark every manifest of that run git_dirty. Write it outside the "
        "repository, or wait for the run to finish.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - a broken guard must not block all work
        print(f"guard_run_lock failed: {exc!r}", file=sys.stderr)
        raise SystemExit(1) from exc
