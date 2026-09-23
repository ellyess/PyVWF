"""PreToolUse guard: a commit that stages code a local-only suite covers needs its stamp.

The suites are defined in `scripts/dev/stamp.py`: `realdata` for the real-data
pins and `pinn` for the physics-informed tests, neither of which CI can run.
If `git commit` would record a change to a suite's covered code, the staged
content must match the fingerprint in that suite's stamp, and that run must
have passed. `git commit -a` is blocked outright, because it stages at commit
time and the paths should be named anyway. `docs/design/agent-guards.md` says
why.

Exit 2 blocks the tool call and shows stderr to Claude. Any unexpected error
exits 1, which lets the call through and shows the error to the user.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path

GIT_COMMIT = re.compile(r"\bgit\b(?:\s+-C\s+\S+)?\s+commit\b")


def _all_flag(command: str) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    for tok in tokens:
        if tok == "--all":
            return True
        if tok.startswith("-") and not tok.startswith("--") and "a" in tok[1:]:
            return True
    return False


def _problem(root: Path, name: str, suite, stamp_path, index_fingerprint, staged) -> str | None:
    rerun = f"Run `python scripts/dev/stamp.py {name}`, read its output, then commit."
    changed = ", ".join(staged)
    path = root / stamp_path(name)
    if not path.exists():
        return f"this commit changes {changed}, and no {name} stamp exists. {rerun}"
    stamp = json.loads(path.read_text(encoding="utf-8"))
    if stamp.get("fingerprint") != index_fingerprint(root, suite.covered):
        return (
            f"this commit changes {changed}, and the staged content is not what the "
            f"last {name} run tested. {rerun}"
        )
    if stamp.get("exit_code") != 0:
        extra = (
            " A pin that moved is re-recorded in this commit, with the size of the "
            "movement in the message."
            if name == "realdata"
            else ""
        )
        return (
            f"the {name} run on this content did not pass ({stamp.get('summary')}).{extra} {rerun}"
        )
    return None


def main() -> int:
    event = json.load(sys.stdin)
    command = (event.get("tool_input") or {}).get("command", "")
    if not GIT_COMMIT.search(command):
        return 0
    if _all_flag(command):
        print(
            "Blocked: `git commit -a` stages at commit time. Read `git status`, "
            "`git add` the paths you mean, then commit.",
            file=sys.stderr,
        )
        return 2

    root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or event.get("cwd") or ".").resolve()
    sys.path.insert(0, str(root / "scripts" / "dev"))
    from stamp import SUITES, index_fingerprint, stamp_path, staged_covered_paths

    problems = []
    for name, suite in SUITES.items():
        staged = staged_covered_paths(root, suite.covered)
        if staged:
            problem = _problem(root, name, suite, stamp_path, index_fingerprint, staged)
            if problem:
                problems.append(problem)
    if problems:
        print("Blocked: " + "\nBlocked: ".join(problems), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - a broken guard must not block all work
        print(f"guard_local_suites failed: {exc!r}", file=sys.stderr)
        raise SystemExit(1) from exc
