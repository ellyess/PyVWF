"""PreToolUse guard: no edits to the tree while a run holds the lock.

A run started through `scripts/dev/run_locked.py` holds `output/.run-lock`.
While it does, this hook blocks Write, Edit, MultiEdit and NotebookEdit on any
path inside the repository that git does not ignore, the git commands that
change the tree, and shell writes to such a path: a `>`, `>>` or `&>`
redirect, `tee` and `sed -i`. A target holding a variable, a path after a `cd`,
and writes by other programs (`cp`, `mv`, `python -c`, a script) are not
caught; the rule in AGENTS.md still covers them. `docs/design/agent-guards.md`
says why.

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

from guard_local_suites import shell_tokens

GIT_WRITES = re.compile(
    r"\bgit\b[^;&|]*?\b(add|am|apply|checkout|cherry-pick|clean|commit|merge|mv|pull"
    r"|rebase|reset|restore|revert|rm|stash|switch)\b"
)

REDIRECTS = {">", ">>", ">|", "&>", "&>>"}
COMMAND_SEPARATORS = {";", "&", "&&", "|", "||", "(", ")", "|&", ";;"}


def shell_write_targets(command: str) -> list[str] | None:
    """Paths a shell command writes by redirect, `tee` or `sed -i`; None if unparseable."""
    try:
        tokens = shell_tokens(command)
    except ValueError:
        return None
    targets, words = [], []
    segments = [words]
    for i, token in enumerate(tokens):
        if token in REDIRECTS:
            if i + 1 < len(tokens):
                targets.append(tokens[i + 1])
        elif token in COMMAND_SEPARATORS:
            words = []
            segments.append(words)
        elif not (i and tokens[i - 1] in REDIRECTS):
            words.append(token)
    for words in segments:
        while words and "=" in words[0] and not words[0].startswith("-"):
            words = words[1:]  # VAR=value prefixes
        if not words:
            continue
        if words[0] == "tee":
            targets += [w for w in words[1:] if not w.startswith("-")]
        elif words[0] == "sed" and any(w.startswith(("-i", "--in-place")) for w in words):
            targets += sed_files(words[1:])
    return targets


def sed_files(args: list[str]) -> list[str]:
    """The files a `sed` argument list names: every positional after the script."""
    files, scripted, skip = [], False, False
    for w in args:
        if skip:
            skip = False
        elif w in ("-e", "-f"):
            scripted, skip = True, True
        elif w.startswith(("--expression", "--file")):
            scripted = True
        elif w and not w.startswith("-"):
            files.append(w)
    return files if scripted else files[1:]


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
        cwd = Path(event.get("cwd") or root)
        for target in shell_write_targets(command) or []:
            if "$" in target or target == "/dev/null":
                continue
            path = Path(os.path.expanduser(target))
            path = (path if path.is_absolute() else cwd / path).resolve()
            if path.is_relative_to(root) and not _ignored(root, path):
                print(
                    f"Blocked: `{running}` holds the run lock. This command writes "
                    f"{path.relative_to(root)}, which would mark every manifest of that "
                    "run git_dirty. Write outside the repository, or wait for the run.",
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
