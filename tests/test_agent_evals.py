"""`scripts/dev/agent_evals/evals.py` scores the critics the way it was registered.

The scoring rule and the pass bar are fixed in
`docs/findings/agent-critics-prereg.md`. Each piece is checked both ways: the
parsers on replies that should and should not count, the blinding check on a
proposal that leaks and on the real ones, and the bar on either side of its
boundary.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from support import load_script

ROOT = Path(__file__).resolve().parents[1]
evals = load_script(ROOT / "scripts" / "dev" / "agent_evals" / "evals.py", "agent_evals")

REVIEW = """**Verdict:** REVISE

**Blockers:**
- The router counts distinct `obs` values with `nunique`, but each cluster's
  observation is a capacity-weighted mean, so rounding makes them differ.

Files read:
/tmp/pyvwf-checkouts/abcd1234/src/vwf/data.py
/tmp/pyvwf-checkouts/abcd1234/AGENTS.md
"""


# ---------------------------------------------------------------- the cases


def test_every_pair_loads_with_both_arms_and_its_key():
    cases = evals.load_cases()
    assert len(cases) == 5
    for case in cases:
        for arm in evals.ARMS:
            assert len(evals.proposal(case, arm).split()) <= 400


def test_the_two_arms_of_a_pair_differ_but_share_their_claim():
    for case in evals.load_cases():
        defect, control = (evals.proposal(case, arm) for arm in evals.ARMS)
        assert defect != control
        assert defect.split("\n\n")[0] == control.split("\n\n")[0], case["pair"]


def test_no_proposal_carries_a_tell():
    cases = evals.load_cases()
    names = [c["pair"] for c in cases]
    for case in cases:
        for arm in evals.ARMS:
            assert evals.blinding_problems(evals.proposal(case, arm), names) == [], (
                case["pair"],
                arm,
            )


def test_the_blinding_check_fires_on_a_proposal_that_leaks():
    text = "As fixed in 26e0b67, see docs/findings/scorecard.md; the ftol-stall eval."
    assert evals.blinding_problems(text, ["ftol-stall"]) == [
        "26e0b67",
        "docs/",
        "eval",
        "ftol-stall",
    ]


def test_a_number_or_a_repository_term_is_not_taken_for_a_tell():
    text = "RMSE 0.1046 over 2015 to 2019, a deadbeef, the evaluate run, run_evaluate"
    assert evals.blinding_problems(text, []) == []


def test_a_pair_missing_its_key_is_refused(tmp_path):
    (tmp_path / "x.toml").write_text('pair = "x"\n')
    with pytest.raises(ValueError, match="lacks"):
        evals.load_cases(tmp_path)


# ---------------------------------------------------------------- prompts


def test_the_critic_prompt_gives_the_proposal_and_the_checkout_and_nothing_else():
    case = evals.load_cases()[0]
    prompt = evals.critic_prompt(evals.proposal(case, "defect"), Path("/tmp/co/1a2b3c4d"))
    assert evals.proposal(case, "defect") in prompt
    assert "/tmp/co/1a2b3c4d" in prompt
    assert evals.blinding_problems(prompt.replace("/tmp/co/1a2b3c4d", ""), [case["pair"]]) == []


def test_the_judge_prompt_names_neither_the_critic_nor_the_arm_nor_the_files():
    prompt = evals.judge_prompt("the router counts floats", REVIEW)
    for word in ("method-critic", "implementation-critic", "red-team", "control", "Files read"):
        assert word not in prompt
    assert "capacity-weighted mean" in prompt


def test_tokens_are_stable_and_distinct():
    assert evals.token("r", "a", "defect", "red-team") == evals.token(
        "r", "a", "defect", "red-team"
    )
    names = {evals.token("r", p, a, c) for p in "abcde" for a in evals.ARMS for c in evals.CRITICS}
    assert len(names) == 30


# ---------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "text,expected",
    [
        ("**Verdict:** REVISE\n", "REVISE"),
        ("Verdict: reject, because", "REJECT"),
        ("**Verdict: PROCEED**", "PROCEED"),
        ("**Verdict:** REVISE. Later: Verdict: REJECT", "REVISE"),
        ("I would revise this.", None),
        ("", None),
    ],
)
def test_the_verdict_is_read_or_reported_missing(text, expected):
    assert evals.parse_verdict(text) == expected


def test_files_read_are_listed_and_checked_against_the_checkout():
    paths = evals.parse_files_read(REVIEW)
    assert paths == [
        "/tmp/pyvwf-checkouts/abcd1234/src/vwf/data.py",
        "/tmp/pyvwf-checkouts/abcd1234/AGENTS.md",
    ]
    assert evals.leaked(paths, "/tmp/pyvwf-checkouts/abcd1234") == []
    outside = paths + ["/repo/scripts/dev/agent_evals/cases/obs-routing.toml"]
    assert evals.leaked(outside, "/tmp/pyvwf-checkouts/abcd1234") == [outside[-1]]
    # A sibling directory sharing the prefix is outside too.
    assert evals.leaked(["/tmp/pyvwf-checkouts/abcd12345/x"], "/tmp/pyvwf-checkouts/abcd1234") != []


def test_a_review_without_a_files_list_is_reported_as_such():
    assert evals.parse_files_read("**Verdict:** PROCEED") is None
    assert evals.leaked(None, "/tmp/x") is None


def test_a_yes_counts_with_a_quote_the_review_contains():
    answer = (
        "FOUND: YES\nQUOTE: each cluster's\n  observation is a capacity-weighted mean, so rounding"
    )
    judged = evals.parse_judgement(answer, REVIEW)
    assert judged == {**judged, "found": True, "said_yes": True, "quote_ok": True}


def test_markdown_and_line_breaks_do_not_break_a_true_quote():
    answer = "FOUND: YES\nQUOTE: counts distinct obs values with nunique"
    assert evals.parse_judgement(answer, REVIEW)["found"]


def test_a_yes_with_an_invented_quote_does_not_count():
    answer = "FOUND: YES\nQUOTE: the tolerance acts as an absolute threshold"
    judged = evals.parse_judgement(answer, REVIEW)
    assert judged["said_yes"] and not judged["quote_ok"] and not judged["found"]


def test_a_yes_with_a_too_short_quote_does_not_count():
    assert not evals.parse_judgement("FOUND: YES\nQUOTE: nunique", REVIEW)["found"]


def test_a_quote_from_the_files_list_does_not_count():
    answer = "FOUND: YES\nQUOTE: /tmp/pyvwf-checkouts/abcd1234/src/vwf/data.py"
    assert not evals.parse_judgement(answer, REVIEW)["found"]


@pytest.mark.parametrize(
    "answer", ["FOUND: NO\nQUOTE: NONE", "FOUND: NO\nQUOTE: counts distinct obs values", ""]
)
def test_a_no_or_no_answer_is_not_found(answer):
    assert not evals.parse_judgement(answer, REVIEW)["found"]


# ---------------------------------------------------------------- outcomes


@pytest.mark.parametrize(
    "arm,verdict,found,excluded,expected",
    [
        ("defect", "REVISE", True, False, "hit"),
        ("defect", "REJECT", True, False, "hit"),
        ("defect", "PROCEED", True, False, "noted"),
        ("defect", None, True, False, "noted"),
        ("defect", "REJECT", False, False, "miss"),
        ("control", "REJECT", True, False, "false alarm"),
        ("control", "PROCEED", True, False, "false alarm"),
        ("control", "REJECT", False, False, "clear"),
        ("defect", "REVISE", True, True, "excluded"),
        ("control", "REVISE", True, True, "excluded"),
    ],
)
def test_each_review_gets_its_registered_outcome(arm, verdict, found, excluded, expected):
    assert evals.outcome(arm, verdict, found, excluded) == expected


def _rows(hit_pairs: set[str], alarm_pairs: set[str], pairs: list[str]) -> list[dict]:
    rows = []
    for pair in pairs:
        for critic in evals.CRITICS:
            hit = pair in hit_pairs and critic == "implementation-critic"
            alarm = pair in alarm_pairs and critic == "red-team"
            rows.append(
                {
                    "pair": pair,
                    "arm": "defect",
                    "critic": critic,
                    "verdict": "REVISE",
                    "outcome": "hit" if hit else "miss",
                }
            )
            rows.append(
                {
                    "pair": pair,
                    "arm": "control",
                    "critic": critic,
                    "verdict": "PROCEED",
                    "outcome": "false alarm" if alarm else "clear",
                }
            )
    return rows


PAIRS = ["a", "b", "c", "d", "e"]


def test_the_bar_passes_at_four_caught_and_one_false_alarm():
    result = evals.tally(_rows({"a", "b", "c", "d"}, {"e"}, PAIRS), PAIRS)
    assert (result["panel_hits"], result["panel_false_alarms"], result["passes"]) == (4, 1, True)
    assert result["per_critic"]["implementation-critic"]["hits"] == 4
    assert result["per_critic"]["red-team"]["false alarms"] == 1


def test_the_bar_fails_at_three_caught():
    result = evals.tally(_rows({"a", "b", "c"}, set(), PAIRS), PAIRS)
    assert (result["panel_hits"], result["passes"]) == (3, False)


def test_the_bar_fails_at_two_false_alarms():
    result = evals.tally(_rows(set(PAIRS), {"a", "b"}, PAIRS), PAIRS)
    assert (result["panel_hits"], result["panel_false_alarms"], result["passes"]) == (5, 2, False)


def test_a_pair_counts_once_however_many_critics_catch_it():
    rows = _rows({"a"}, set(), PAIRS)
    for r in rows:
        if r["pair"] == "a" and r["arm"] == "defect":
            r["outcome"] = "hit"
    assert evals.tally(rows, PAIRS)["panel_hits"] == 1


def test_blocking_verdicts_are_counted_per_arm():
    counts = evals.tally(_rows(set(), set(), PAIRS), PAIRS)["per_critic"]["method-critic"]
    assert counts["blocking verdicts, defect arm"] == 5
    assert counts["blocking verdicts, control arm"] == 0


# ------------------------------------------------------------ end to end


def _scored_run(tmp_path, monkeypatch, calibration_answer):
    """A run of two reviews, judged, with every calibration answered the same way."""
    monkeypatch.setattr(evals, "RUNS", tmp_path)
    run_dir = tmp_path / "t"
    (run_dir / "responses").mkdir(parents=True)
    (run_dir / "judgements").mkdir()
    snapshot = "/tmp/pyvwf-checkouts/abcd1234"
    tasks = [
        {"token": "t1", "pair": "obs-routing", "arm": "defect", "critic": "implementation-critic"},
        {"token": "t2", "pair": "obs-routing", "arm": "control", "critic": "red-team"},
    ]
    record = {
        "label": "t",
        "critic_model": "m",
        "judge_model": "j",
        "harness_commit": "0" * 40,
        "snapshots": {"obs-routing": {"dir": snapshot}},
        "tasks": tasks,
    }
    (run_dir / "run.json").write_text(evals.json.dumps(record))
    (run_dir / "responses" / "t1.md").write_text(REVIEW)
    (run_dir / "responses" / "t2.md").write_text(
        "**Verdict:** PROCEED\n\nFiles read:\n/elsewhere/AGENTS.md\n"
    )
    evals.judge(run_dir, "t")
    record = evals.json.loads((run_dir / "run.json").read_text())
    for cal in record["calibration"]:
        fixture = evals.load_cases(evals.CASES)
        text = {c["pair"]: c for c in fixture}[cal["pair"]][f"judge_{cal['kind']}"]
        answer = calibration_answer(cal["kind"], text)
        (run_dir / "judgements" / f"{cal['token']}.md").write_text(answer)
    (run_dir / "judgements" / "t1.md").write_text(
        "FOUND: YES\nQUOTE: counts distinct `obs` values with `nunique`"
    )
    (run_dir / "judgements" / "t2.md").write_text("FOUND: NO\nQUOTE: NONE")
    return evals.score(run_dir)


def _honest(kind, text):
    if kind == "positive":
        return "FOUND: YES\nQUOTE: " + " ".join(text.split())[20:80]
    return "FOUND: NO\nQUOTE: NONE"


def test_a_scored_run_records_each_review_and_reads_the_bar(tmp_path, monkeypatch):
    result = _scored_run(tmp_path, monkeypatch, _honest)
    assert result["judge_calibrated"]
    t1, t2 = result["rows"]
    assert (t1["verdict"], t1["found"], t1["files_read"], t1["outcome"]) == (
        "REVISE",
        True,
        "inside",
        "hit",
    )
    assert (t2["files_read"], t2["outcome"]) == ("outside snapshot", "excluded")
    assert t2["outside"] == ["/elsewhere/AGENTS.md"]
    assert result["panel"]["obs-routing"] == {"defect caught": True, "control false alarm": False}
    assert result["passes"] is False  # one pair caught of five
    assert (tmp_path / "t" / "results.md").read_text().startswith("Run t:")


def test_a_judge_that_fails_calibration_leaves_the_bar_unread(tmp_path, monkeypatch):
    result = _scored_run(tmp_path, monkeypatch, lambda kind, text: _honest("positive", text))
    assert not result["judge_calibrated"]
    assert result["passes"] is None
    assert "not read" in (tmp_path / "t" / "results.md").read_text()


def test_judging_is_refused_while_a_review_is_missing(tmp_path, monkeypatch):
    run_dir = tmp_path / "t"
    run_dir.mkdir()
    task = {"token": "t1", "pair": "obs-routing", "arm": "defect", "critic": "red-team"}
    (run_dir / "run.json").write_text(evals.json.dumps({"tasks": [task]}))
    with pytest.raises(SystemExit, match="1 responses missing"):
        evals.judge(run_dir, "t")
