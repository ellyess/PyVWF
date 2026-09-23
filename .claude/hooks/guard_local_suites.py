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
HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1")
SEPARATORS = {";", "&", "&&", "|", "||", "(", ")", "<", ">", "<<", ">>", "<<<", "|&", ";;"}
# Short commit options whose value follows, attached or as the next token.
SHORT_WITH_VALUE = set("mFCct")
LONG_WITH_VALUE = {
    "--message",
    "--file",
    "--reuse-message",
    "--reedit-message",
    "--template",
    "--author",
    "--date",
    "--fixup",
    "--squash",
    "--cleanup",
    "--trailer",
    "--pathspec-from-file",
}


def _strip_heredocs(command: str) -> str:
    """Drop heredoc bodies, which are data, so a message cannot look like a command."""
    lines = command.split("\n")
    out, delimiter = [], None
    for line in lines:
        if delimiter is not None:
            if line.strip() == delimiter:
                delimiter = None
            continue
        out.append(line)
        found = HEREDOC.findall(line)
        if found:
            delimiter = found[-1][1]
    return "\n".join(out)


def _tokens(command: str) -> list[str]:
    """Shell tokens, with newlines as separators; raises ValueError on bad quoting."""
    lexer = shlex.shlex(
        _strip_heredocs(command).replace("\n", " ; "), posix=True, punctuation_chars=True
    )
    lexer.whitespace_split = True
    return list(lexer)


def commit_invocations(command: str) -> list[list[str]] | None:
    """The argument lists of every `git commit` run by the command, or None if unparseable.

    Only a `git` in command position counts: at the start or after a separator,
    past any VAR=value assignments. Git's own options before `commit` are
    skipped. Text inside quotes or heredocs is never read as a command.
    """
    try:
        tokens = _tokens(command)
    except ValueError:
        return None
    found: list[list[str]] = []
    at_start, i = True, 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in SEPARATORS:
            at_start, i = True, i + 1
            continue
        if at_start and "=" in tok and not tok.startswith("-"):
            i += 1
            continue
        if at_start and Path(tok).name == "git":
            j = i + 1
            while j < len(tokens) and tokens[j].startswith("-"):
                j += 2 if tokens[j] in ("-C", "-c") else 1
            if j < len(tokens) and tokens[j] == "commit":
                k = j + 1
                while k < len(tokens) and tokens[k] not in SEPARATORS:
                    k += 1
                found.append(tokens[j + 1 : k])
        at_start, i = False, i + 1
    return found


def stages_all(args: list[str]) -> bool:
    """Whether a `git commit` argument list includes -a or --all."""
    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "--all":
            return True
        if tok in LONG_WITH_VALUE:
            i += 2
            continue
        if tok.startswith("-") and not tok.startswith("--"):
            for pos, char in enumerate(tok[1:], start=1):
                if char == "a":
                    return True
                if char in SHORT_WITH_VALUE:
                    if pos == len(tok) - 1:
                        i += 1  # the value is the next token
                    break
        i += 1
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
    invocations = commit_invocations(command) or []
    # A commit the parser cannot see, such as one inside `sh -c`, still gets
    # the staged-code check through the plain text match; only a parsed
    # invocation can be refused for -a, so a message cannot trip that.
    if not invocations and not GIT_COMMIT.search(command):
        return 0
    if invocations and any(stages_all(args) for args in invocations):
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
