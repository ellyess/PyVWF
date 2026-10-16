"""Measure whether the review agents catch defects this repository has caught.

The cases, the scoring rule and the pass bar are pre-registered in
`docs/findings/agent-critics-prereg.md`; this script carries them out and
changes none of them. Each case in `cases/` is a matched pair: a proposal
carrying a defect the repository found and fixed, and the same proposal with
the defect removed. A critic reviews one proposal against a checkout of the
repository as it stood before the fix, so the fix and its write-up are not
there to find.

The critics are Claude Code subagents (`.claude/agents/`), so a Claude Code
session makes the agent calls, the way `idea-review` does; this script writes
what each call is given and scores what comes back. Three steps:

    python scripts/dev/agent_evals/evals.py prepare --label 2026-10-16 \\
        --critic-model claude-opus-5-5 --judge-model claude-sonnet-5-5
    # the session sends prompts/<token>.md to the critic run.json names for it,
    # and saves each reply, unedited, as responses/<token>.md
    python scripts/dev/agent_evals/evals.py judge --label 2026-10-16
    # the session gives each judge_prompts/<token>.md to the judge model, and
    # saves each answer as judgements/<token>.md
    python scripts/dev/agent_evals/evals.py score --label 2026-10-16

Nothing here touches the network; the agent calls are the session's.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import tarfile
import tempfile
from datetime import date
from pathlib import Path

try:  # tomllib entered the stdlib in 3.11; 3.10 uses the tomli backport.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CASES = HERE / "cases"
RUNS = HERE / "runs"
SNAPSHOT_ROOT = Path(tempfile.gettempdir()) / "pyvwf-checkouts"

CRITICS = ("method-critic", "implementation-critic", "red-team")
ARMS = ("defect", "control")
VERDICTS = ("PROCEED", "REVISE", "REJECT")
BLOCKING = frozenset({"REVISE", "REJECT"})
# A judge's quote shorter than this cannot carry a mechanism, so it does not
# count as evidence that the review found the defect.
MIN_QUOTE = 15
# The pass bar, fixed in the pre-registration: the panel of three critics
# catches at least this many of the five defects ...
PANEL_HITS_TO_PASS = 4
# ... and raises the paired defect on at most this many of the five controls.
PANEL_FALSE_ALARMS_ALLOWED = 1

PAIR_FIELDS = ("pair", "snapshot", "source_commit", "source_document", "expected_catchers")
KEY_FIELDS = ("defect", "judge_positive", "judge_negative")

# What a proposal must not contain, so that nothing in it points at the fix or
# at the fact that it is a test: a commit hash, a document path, and words a
# real proposal would not use about itself.
HEX = re.compile(r"\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b")
TELLS = re.compile(
    r"docs/|correction notice|retract|defect|seeded|blind|\bevals?\b|critic|agent_evals|"
    r"\bstale\b|\bwrong\b|known bug",
    re.IGNORECASE,
)

VERDICT = re.compile(r"\*{0,2}Verdict:?\*{0,2}:?\s*\*{0,2}\s*(PROCEED|REVISE|REJECT)\b", re.I)
FILES_READ = re.compile(r"^\s*\**Files read:?\**:?\s*$", re.IGNORECASE | re.MULTILINE)
FOUND = re.compile(r"^\s*FOUND:\s*(YES|NO)\b", re.IGNORECASE | re.MULTILINE)
QUOTE = re.compile(r"^\s*QUOTE:\s*(.*)\Z", re.IGNORECASE | re.MULTILINE | re.DOTALL)

CRITIC_PROMPT = """\
Review this proposal for PyVWF.

The repository is checked out at {snapshot}. Treat that directory as the
repository root: resolve every path in your instructions and in the proposal
against it, and read nothing outside it.

After your review, end your reply with a line "Files read:" followed by the
absolute path of every file you opened, one per line.

## Proposal

{proposal}
"""

JUDGE_PROMPT = """\
You are grading one review of a proposal. You do not see the proposal. Below
is a description of one specific problem, then the review. Do not open any
file; judge from the text here alone.

Question: does the review identify this problem as something wrong or missing
in the proposal? Count it only if the review states the problem's mechanism,
in its own words, as a fault of the proposal. A review that mentions the same
code or setting without saying it is at fault does not count. A review that
credits the proposal with handling it does not count.

Answer in exactly two lines:
FOUND: YES or FOUND: NO
QUOTE: the shortest passage from the review that states the problem, copied
character for character; or NONE

## The problem

{defect}

## The review

{review}
"""


# ---------------------------------------------------------------- the cases


def load_cases(cases_dir: Path = CASES) -> list[dict]:
    """Read every pair, checking each carries the fields the run needs."""
    pairs = []
    for path in sorted(cases_dir.glob("*.toml")):
        with path.open("rb") as fh:
            case = tomllib.load(fh)
        missing = [f for f in PAIR_FIELDS + KEY_FIELDS if not case.get(f)]
        missing += [
            f"{arm}_arm.proposal" for arm in ARMS if not case.get(f"{arm}_arm", {}).get("proposal")
        ]
        if missing:
            raise ValueError(f"{path.name} lacks {', '.join(missing)}")
        if case["pair"] != path.stem:
            raise ValueError(f"{path.name} names its pair {case['pair']!r}")
        unknown = set(case["expected_catchers"]) - set(CRITICS)
        if unknown:
            raise ValueError(f"{path.name} expects unknown critics {sorted(unknown)}")
        pairs.append(case)
    return pairs


def proposal(case: dict, arm: str) -> str:
    return case[f"{arm}_arm"]["proposal"].strip()


def blinding_problems(text: str, pair_names: list[str]) -> list[str]:
    """What in a proposal could tell a critic which case it is, or that it is one."""
    found = [m.group(0) for m in HEX.finditer(text)]
    found += [m.group(0) for m in TELLS.finditer(text)]
    found += [name for name in pair_names if name in text]
    return found


# ----------------------------------------------------------------- the run


def token(label: str, *parts: str) -> str:
    """An opaque, stable name for one prompt, so file names say nothing."""
    return hashlib.sha256("|".join((label, *parts)).encode()).hexdigest()[:10]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_snapshot(sha: str, dest: Path) -> None:
    """Write the tracked tree at `sha` to `dest`, with no git metadata."""
    tar_bytes = subprocess.run(
        ["git", "archive", "--format=tar", sha], cwd=ROOT, capture_output=True, check=True
    ).stdout
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
        else:  # pragma: no cover - Pythons without extraction filters
            tar.extractall(dest)  # noqa: S202 - our own repository's archive


def critic_prompt(text: str, snapshot: Path) -> str:
    return CRITIC_PROMPT.format(snapshot=snapshot, proposal=text)


def strip_files_read(review: str) -> str:
    """The review without its closing list of files, which the judge does not need."""
    match = FILES_READ.search(review)
    return review[: match.start()].rstrip() if match else review.rstrip()


def judge_prompt(defect: str, review: str) -> str:
    return JUDGE_PROMPT.format(defect=defect.strip(), review=strip_files_read(review))


def prepare(
    run_dir: Path, label: str, critic_model: str, judge_model: str, snapshot_root: Path
) -> dict:
    """Extract the snapshots and write one prompt per (case, arm, critic)."""
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists; a run is prepared once")
    pairs = load_cases()
    names = [c["pair"] for c in pairs]
    for case in pairs:
        for arm in ARMS:
            tells = blinding_problems(proposal(case, arm), names)
            if tells:
                raise SystemExit(f"{case['pair']} {arm} proposal is not blind: {tells}")
    (run_dir / "prompts").mkdir(parents=True)
    snapshots, tasks = {}, []
    for case in pairs:
        sha = _git("rev-parse", case["snapshot"])
        where = snapshot_root / hashlib.sha256(sha.encode()).hexdigest()[:8]
        if not where.exists():
            extract_snapshot(sha, where)
        snapshots[case["pair"]] = {"rev": case["snapshot"], "sha": sha, "dir": str(where)}
        for arm in ARMS:
            for critic in CRITICS:
                name = token(label, case["pair"], arm, critic)
                (run_dir / "prompts" / f"{name}.md").write_text(
                    critic_prompt(proposal(case, arm), where)
                )
                tasks.append({"token": name, "pair": case["pair"], "arm": arm, "critic": critic})
    tasks.sort(key=lambda t: t["token"])
    record = {
        "label": label,
        "prepared": date.today().isoformat(),
        "critic_model": critic_model,
        "judge_model": judge_model,
        "harness_commit": _git("rev-parse", "HEAD"),
        "harness_dirty": bool(_git("status", "--porcelain")),
        "critic_definitions": {
            c: _sha256(ROOT / ".claude" / "agents" / f"{c}.md") for c in CRITICS
        },
        "snapshots": snapshots,
        "tasks": tasks,
    }
    (run_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{len(tasks)} prompts in {run_dir / 'prompts'}; send each to the critic run.json names")
    return record


def judge(run_dir: Path, label: str) -> list[str]:
    """Write one judge prompt per review, and two calibration prompts per pair."""
    record = json.loads((run_dir / "run.json").read_text())
    pairs = {c["pair"]: c for c in load_cases()}
    missing = [
        t["token"]
        for t in record["tasks"]
        if not (run_dir / "responses" / f"{t['token']}.md").exists()
    ]
    if missing:
        raise SystemExit(f"{len(missing)} responses missing, first {missing[0]}")
    out = run_dir / "judge_prompts"
    out.mkdir(exist_ok=True)
    written = []
    for t in record["tasks"]:
        review = (run_dir / "responses" / f"{t['token']}.md").read_text()
        (out / f"{t['token']}.md").write_text(judge_prompt(pairs[t["pair"]]["defect"], review))
        written.append(t["token"])
    calibration = []
    for name, case in pairs.items():
        for kind in ("positive", "negative"):
            cal = token(label, "calibration", name, kind)
            (out / f"{cal}.md").write_text(judge_prompt(case["defect"], case[f"judge_{kind}"]))
            calibration.append({"token": cal, "pair": name, "kind": kind})
            written.append(cal)
    record["calibration"] = calibration
    (run_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{len(written)} judge prompts in {out}")
    return written


# ------------------------------------------------------------- the scoring


def parse_verdict(review: str) -> str | None:
    """The first verdict the review states, or None if it states none."""
    match = VERDICT.search(review)
    return match.group(1).upper() if match else None


def parse_files_read(review: str) -> list[str] | None:
    """The paths listed after "Files read:", or None if there is no such list."""
    match = FILES_READ.search(review)
    if not match:
        return None
    lines = review[match.end() :].splitlines()
    return [ln.strip().lstrip("-* ").strip("`") for ln in lines if ln.strip()]


def leaked(paths: list[str] | None, snapshot: str) -> list[str] | None:
    """Listed paths outside the snapshot; None when the critic listed nothing."""
    if paths is None:
        return None
    root = snapshot.rstrip("/") + "/"
    return [p for p in paths if not (p + "/").startswith(root)]


def _normalise(text: str) -> str:
    return " ".join(text.replace("*", "").replace("`", "").split()).lower()


def parse_judgement(answer: str, review: str) -> dict:
    """Read a judge's two lines; a YES counts only with a quote the review contains.

    Returns found (the defect counts as identified), said_yes (the judge's own
    answer), and quote_ok (the quote is at least MIN_QUOTE characters and is in
    the review, ignoring whitespace, case, asterisks and backticks).
    """
    found = FOUND.search(answer)
    said_yes = bool(found and found.group(1).upper() == "YES")
    quote_match = QUOTE.search(answer)
    quote = quote_match.group(1).strip().strip('"') if quote_match else ""
    quote_ok = (
        quote.upper() != "NONE"
        and len(_normalise(quote)) >= MIN_QUOTE
        and _normalise(quote) in _normalise(strip_files_read(review))
    )
    return {
        "found": said_yes and quote_ok,
        "said_yes": said_yes,
        "quote_ok": quote_ok,
        "quote": quote,
    }


def outcome(arm: str, verdict: str | None, found: bool, excluded: bool) -> str:
    """The pre-registered outcome of one review."""
    if excluded:
        return "excluded"
    if arm == "defect":
        if found and verdict in BLOCKING:
            return "hit"
        return "noted" if found else "miss"
    return "false alarm" if found else "clear"


def tally(rows: list[dict], pairs: list[str]) -> dict:
    """Per-critic counts, the panel's per-pair result, and the pass bar."""
    per_critic = {}
    for critic in CRITICS:
        mine = [r for r in rows if r["critic"] == critic]
        per_critic[critic] = {
            "hits": sum(r["outcome"] == "hit" for r in mine),
            "noted": sum(r["outcome"] == "noted" for r in mine),
            "misses": sum(r["outcome"] == "miss" for r in mine),
            "false alarms": sum(r["outcome"] == "false alarm" for r in mine),
            "clear": sum(r["outcome"] == "clear" for r in mine),
            "excluded": sum(r["outcome"] == "excluded" for r in mine),
            "blocking verdicts, defect arm": sum(
                r["verdict"] in BLOCKING for r in mine if r["arm"] == "defect"
            ),
            "blocking verdicts, control arm": sum(
                r["verdict"] in BLOCKING for r in mine if r["arm"] == "control"
            ),
        }
    panel = {}
    for pair in pairs:
        defect = [r["outcome"] for r in rows if r["pair"] == pair and r["arm"] == "defect"]
        control = [r["outcome"] for r in rows if r["pair"] == pair and r["arm"] == "control"]
        panel[pair] = {
            "defect caught": "hit" in defect,
            "control false alarm": "false alarm" in control,
        }
    hits = sum(p["defect caught"] for p in panel.values())
    alarms = sum(p["control false alarm"] for p in panel.values())
    return {
        "per_critic": per_critic,
        "panel": panel,
        "panel_hits": hits,
        "panel_false_alarms": alarms,
        "passes": hits >= PANEL_HITS_TO_PASS and alarms <= PANEL_FALSE_ALARMS_ALLOWED,
    }


def score(run_dir: Path) -> dict:
    """Score every review, check the judge's calibration, and apply the bar."""
    record = json.loads((run_dir / "run.json").read_text())
    if "calibration" not in record:
        raise SystemExit("run judge first")
    pairs = {c["pair"]: c for c in load_cases()}

    def answer(name: str) -> str:
        path = run_dir / "judgements" / f"{name}.md"
        if not path.exists():
            raise SystemExit(f"judgement missing: {path}")
        return path.read_text()

    calibration = []
    for cal in record["calibration"]:
        fixture = pairs[cal["pair"]][f"judge_{cal['kind']}"]
        judged = parse_judgement(answer(cal["token"]), fixture)
        calibration.append(
            {
                **cal,
                "found": judged["found"],
                "passed": judged["found"] == (cal["kind"] == "positive"),
            }
        )
    calibrated = all(c["passed"] for c in calibration)

    rows = []
    for t in record["tasks"]:
        review = (run_dir / "responses" / f"{t['token']}.md").read_text()
        snapshot = record["snapshots"][t["pair"]]["dir"]
        outside = leaked(parse_files_read(review), snapshot)
        judged = parse_judgement(answer(t["token"]), review)
        verdict = parse_verdict(review)
        excluded = bool(outside)
        rows.append(
            {
                **t,
                "verdict": verdict,
                "said_yes": judged["said_yes"],
                "quote_ok": judged["quote_ok"],
                "found": judged["found"],
                "files_read": "not listed"
                if outside is None
                else ("outside snapshot" if outside else "inside"),
                "outside": outside or [],
                "outcome": outcome(t["arm"], verdict, judged["found"], excluded),
                "quote": judged["quote"],
            }
        )
    rows.sort(key=lambda r: (r["pair"], ARMS.index(r["arm"]), CRITICS.index(r["critic"])))
    result = {
        "label": record["label"],
        "critic_model": record["critic_model"],
        "judge_model": record["judge_model"],
        "harness_commit": record["harness_commit"],
        "calibration": calibration,
        "judge_calibrated": calibrated,
        "rows": rows,
        **tally(rows, sorted(pairs)),
    }
    if not calibrated:
        result["passes"] = None
    (run_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    table = render(result)
    (run_dir / "results.md").write_text(table + "\n")
    print(table)
    return result


def render(result: dict) -> str:
    """The raw table first, then the counts, then the bar."""
    lines = [
        f"Run {result['label']}: critics {result['critic_model']}, judge {result['judge_model']}, "
        f"harness at {result['harness_commit'][:7]}.",
        "",
        "| Pair | Arm | Critic | Verdict | Judge | Quote in review | Files read | Outcome |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in result["rows"]:
        lines.append(
            f"| {r['pair']} | {r['arm']} | {r['critic']} | {r['verdict'] or 'none'} | "
            f"{'YES' if r['said_yes'] else 'NO'} | {'yes' if r['quote_ok'] else 'no'} | "
            f"{r['files_read']} | {r['outcome']} |"
        )
    lines += [
        "",
        "| Critic | Hits | Noted | Misses | False alarms | Clear | Excluded |",
        "|---|---|---|---|---|---|---|",
    ]
    for critic, c in result["per_critic"].items():
        lines.append(
            f"| {critic} | {c['hits']} | {c['noted']} | {c['misses']} | {c['false alarms']} | "
            f"{c['clear']} | {c['excluded']} |"
        )
    cal_failed = [f"{c['pair']} {c['kind']}" for c in result["calibration"] if not c["passed"]]
    lines += [
        "",
        f"Judge calibration: {'passed' if not cal_failed else 'FAILED on ' + ', '.join(cal_failed)}.",
        f"Panel: {result['panel_hits']} of {len(result['panel'])} defects caught, "
        f"{result['panel_false_alarms']} of {len(result['panel'])} controls with a false alarm.",
        "Bar: "
        + {True: "passed", False: "not passed", None: "not read (judge calibration failed)"}[
            result["passes"]
        ]
        + f" (at least {PANEL_HITS_TO_PASS} caught, at most {PANEL_FALSE_ALARMS_ALLOWED} false alarm).",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------- cli


def main(
    command: str,
    label: str,
    critic_model: str | None = None,
    judge_model: str | None = None,
    snapshot_root: Path | None = None,
) -> object:
    run_dir = RUNS / label
    if command == "prepare":
        if not critic_model or not judge_model:
            raise SystemExit(
                "prepare needs --critic-model and --judge-model, recorded with the run"
            )
        return prepare(run_dir, label, critic_model, judge_model, snapshot_root or SNAPSHOT_ROOT)
    if command == "judge":
        return judge(run_dir, label)
    return score(run_dir)


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("prepare", "judge", "score"))
    parser.add_argument("--label", required=True, help="the run's name, its directory under runs/")
    parser.add_argument("--critic-model", help="the model the critics ran on (prepare)")
    parser.add_argument("--judge-model", help="the model the judge ran on (prepare)")
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        help=f"where the checkouts the critics read are written (default {SNAPSHOT_ROOT})",
    )
    args = parser.parse_args(argv)
    main(args.command, args.label, args.critic_model, args.judge_model, args.snapshot_root)


if __name__ == "__main__":
    cli()
