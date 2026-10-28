"""PreToolUse guard for the docs-critic agent: Bash runs only read-only git and wc.

A subagent's `tools` field takes tool names, not command patterns, so this
hook, registered in `.claude/agents/docs-critic.md`, is what limits that
agent's Bash. Every subcommand must be `git diff`, `git show`, `git log` or
`wc`, with no git option before the subcommand (`git -C`, `-c`), no
`--output`, and no redirect. Anything else is blocked.

Unlike the repository's other hooks, this one fails closed: a command it cannot
parse, or an error, blocks the call, because the agent it guards has no other
limit on Bash. `docs/design/ai-native-development.md` says why.
"""

from __future__ import annotations

import json
import sys

from guard_local_suites import shell_tokens

ALLOWED_GIT = {"diff", "show", "log"}
SEPARATORS = {";", "&&", "|", "||"}
REFUSED = {">", ">>", ">|", "&>", "&>>", "<", "<<", "<<<", "&", "(", ")", "|&", ";;"}


def refusal(command: str) -> str | None:
    """Why the command is refused, or None if every subcommand is allowed."""
    try:
        tokens = shell_tokens(command)
    except ValueError:
        return "it cannot be parsed"
    segments, words = [], []
    for token in tokens:
        if token in REFUSED:
            return f"`{token}` is not allowed"
        if token in SEPARATORS:
            segments.append(words)
            words = []
        else:
            words.append(token)
    segments.append(words)
    for words in segments:
        if not words:
            continue
        if any(w.startswith("--output") for w in words):
            return "`--output` writes a file"
        if words[0] == "wc":
            continue
        if words[0] == "git" and len(words) > 1 and words[1] in ALLOWED_GIT:
            continue
        return f"`{' '.join(words[:2])}` is not git diff, show, log or wc"
    return None


def main() -> int:
    event = json.load(sys.stdin)
    command = (event.get("tool_input") or {}).get("command", "")
    why = refusal(command)
    if why is None:
        return 0
    print(
        f"Blocked for docs-critic: {why}. It may run git diff, show, log and wc.", file=sys.stderr
    )
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - this guard fails closed
        print(f"guard_docs_critic failed, so the call is blocked: {exc!r}", file=sys.stderr)
        raise SystemExit(2) from exc
