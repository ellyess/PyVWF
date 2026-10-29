"""`scripts/dev/agent_evals/evals.py` scores the critics the way it was registered.

The scoring rule and the pass bar are fixed in
`docs/findings/agent-critics-prereg.md`, and the docs suite in
`docs/findings/agent-critics-docs-prereg.md`. Each piece is checked both ways: the
parsers on replies that should and should not count, the blinding check on a
proposal that leaks and on the real ones, and the bar on either side of its
boundary.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from support import ROOT, load_script

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
    names = {
        evals.token("r", p, a, c) for p in "abcde" for a in evals.ARMS for c in evals.REVIEWERS
    }
    assert len(names) == 60


# ------------------------------------------------------------- reviewers


def test_a_definition_is_read_as_frontmatter_and_a_body():
    meta, body = evals.read_definition(ROOT / ".claude" / "agents" / "method-critic.md")
    assert (meta["name"], meta["tools"], meta["model"]) == (
        "method-critic",
        "Read, Grep, Glob",
        "opus",
    )
    assert body.startswith("You review") and "---" not in body[:5]


def test_the_agents_json_defines_six_read_only_reviewers_without_claude_md():
    agents, hashes = evals.agents_json()
    assert sorted(agents) == sorted(evals.REVIEWERS)
    assert all(a["omitClaudeMd"] is True for a in agents.values())
    assert all(a["tools"] == ["Read", "Grep", "Glob"] for a in agents.values())
    for baseline, critic in evals.BASELINES.items():
        assert agents[baseline]["model"] == agents[critic]["model"]
        assert agents[baseline]["prompt"] == agents["baseline-1"]["prompt"]
    assert sorted(hashes) == sorted([*evals.CRITICS, "baseline-reviewer"])
    assert all(len(h) == 64 for h in hashes.values())


DOCS = evals.SUITES["docs"]


def test_every_docs_pair_loads_blind_with_both_arms():
    cases = evals.load_cases(DOCS["cases"], DOCS["critics"])
    names = [c["pair"] for c in cases]
    assert names == [
        "cited-test",
        "dropped-caveat",
        "file-count",
        "pinned-halves",
        "reproduction-claim",
    ]
    for case in cases:
        assert evals.proposal(case, "defect") != evals.proposal(case, "control")
        for arm in evals.ARMS:
            assert evals.blinding_problems(evals.proposal(case, arm), names, case["allow"]) == []


def test_a_page_path_is_a_tell_unless_its_case_allows_it():
    case = {c["pair"]: c for c in evals.load_cases(DOCS["cases"], DOCS["critics"])}[
        "dropped-caveat"
    ]
    text = evals.proposal(case, "defect")
    assert evals.blinding_problems(text, [], []) != []
    assert evals.blinding_problems(text, [], case["allow"]) == []


def test_the_docs_baseline_takes_the_critics_model_tools_and_absolute_hook():
    agents, hashes = evals.agents_json("docs")
    critic, baseline = agents["docs-critic"], agents["baseline-docs"]
    assert sorted(agents) == ["baseline-docs", "docs-critic"]
    assert critic["tools"] == baseline["tools"] == ["Read", "Grep", "Glob", "Bash"]
    assert critic["model"] == baseline["model"] == "opus"
    assert critic["hooks"] == baseline["hooks"]
    command = critic["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert str(ROOT) in command and "$CLAUDE_PROJECT_DIR" not in command
    assert critic["prompt"] != baseline["prompt"]
    assert all(a["omitClaudeMd"] is True for a in agents.values())
    assert sorted(hashes) == ["baseline-reviewer", "docs-critic"]


def test_the_proposal_reviewers_carry_no_hooks():
    agents, _ = evals.agents_json()
    assert all("hooks" not in a for a in agents.values())


def test_a_session_directory_must_lie_outside_git(tmp_path):
    assert evals.outside_git(tmp_path)
    assert not evals.outside_git(ROOT)


def test_the_baseline_definition_says_nothing_about_this_repository():
    _, body = evals.read_definition(evals.BASELINE_DEFINITION)
    for word in (
        "pyvwf",
        "harness",
        "scalar",
        "offset",
        "glob",
        "AGENTS",
        "docs/",
        "known positive",
    ):
        assert word.lower() not in body.lower(), word
    assert "Verdict: REVISE" in body


# ---------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "text,expected",
    [
        ("**Verdict:** REVISE\n", "REVISE"),
        ("Verdict: reject, because", "REJECT"),
        ("**Verdict: PROCEED**", "PROCEED"),
        ("**Verdict:** REVISE. Later: Verdict: REJECT", "REVISE"),
        ("**Verdict:** FIX.\nThen one row per finding", "FIX"),
        ("**Verdict:** PASS", "PASS"),
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


# Notes the reviews of 2026-10-29 wrote under their files lists; every one was
# read as a path outside the snapshot, and so excluded its review.
NOTES = """Files read:
/tmp/pyvwf-checkouts/abcd1234/src/vwf/data.py
(docs/CONTEXT.md does not exist at this checkout)
(grep only: /tmp/pyvwf-checkouts/abcd1234/a.py, /tmp/pyvwf-checkouts/abcd1234/b.py)
(Tried to open these, but they don't exist: tests/test_harness.py)

Verdict: REVISE
"""


def test_a_note_under_the_files_list_is_not_a_read_outside():
    paths = evals.parse_files_read(NOTES)
    assert paths == [
        "/tmp/pyvwf-checkouts/abcd1234/src/vwf/data.py",
        "/tmp/pyvwf-checkouts/abcd1234/a.py",
        "/tmp/pyvwf-checkouts/abcd1234/b.py",
    ]
    assert evals.leaked(paths, "/tmp/pyvwf-checkouts/abcd1234") == []


def test_a_note_naming_a_path_outside_the_checkout_is_still_a_leak():
    review = NOTES + "(also grepped /repo/scripts/dev/agent_evals/cases/obs-routing.toml)\n"
    paths = evals.parse_files_read(review)
    assert evals.leaked(paths, "/tmp/pyvwf-checkouts/abcd1234") == [
        "/repo/scripts/dev/agent_evals/cases/obs-routing.toml"
    ]


def test_a_listed_path_with_spaces_is_kept_whole():
    review = "Files read:\n/Volumes/My Disk/repo/AGENTS.md\n"
    outside = evals.leaked(evals.parse_files_read(review), "/tmp/pyvwf-checkouts/abcd1234")
    assert "/Volumes/My Disk/repo/AGENTS.md" in outside


def test_a_second_path_on_a_listed_line_is_checked_too():
    review = "Files read:\n/tmp/pyvwf-checkouts/abcd1234/a.py, /Users/me/repo/AGENTS.md\n"
    assert evals.leaked(evals.parse_files_read(review), "/tmp/pyvwf-checkouts/abcd1234") == [
        "/Users/me/repo/AGENTS.md"
    ]


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
        ("defect", "FIX", True, False, "hit"),
        ("defect", "PASS", True, False, "noted"),
        ("control", "PASS", False, False, "clear"),
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


DOCS_REVIEW = """**Verdict:** FIX. CHANGELOG.md | counts | the count does not recount.

Commands run:
wc -l /tmp/pyvwf-checkouts/abcd1234/CHANGELOG.md
{extra}
Files read:
/tmp/pyvwf-checkouts/abcd1234/CHANGELOG.md
"""
SNAP = "/tmp/pyvwf-checkouts/abcd1234"


def test_commands_inside_the_checkout_are_not_a_leak():
    review = DOCS_REVIEW.format(extra="none")
    assert evals.parse_commands(review) == ["wc -l /tmp/pyvwf-checkouts/abcd1234/CHANGELOG.md"]
    assert evals.paths_outside(evals.parse_commands(review), SNAP) == []


def test_a_command_naming_a_path_outside_the_checkout_is_a_leak():
    review = DOCS_REVIEW.format(extra="wc -l '/Users/me/repo/CHANGELOG.md'")
    assert evals.paths_outside(evals.parse_commands(review), SNAP) == [
        "/Users/me/repo/CHANGELOG.md"
    ]


def test_a_redirect_to_dev_null_is_not_a_read():
    command = "ls /tmp/pyvwf-checkouts/abcd1234/output 2>/dev/null | head"
    assert evals.paths_outside([command], SNAP) == []
    assert evals.paths_outside([command + " /Users/me/repo/x"], SNAP) == ["/Users/me/repo/x"]


def test_the_judge_sees_neither_the_commands_nor_the_files():
    stripped = evals.strip_files_read(DOCS_REVIEW.format(extra="none"))
    assert stripped.endswith("does not recount.") and "wc -l" not in stripped


def test_a_docs_run_reads_one_reviewer_against_one_baseline():
    rows = []
    for name, kind, caught in (
        ("docs-critic", "critic", set(PAIRS)),
        ("baseline-docs", "baseline", {"a", "b", "c"}),
    ):
        for pair in PAIRS:
            rows.append(
                {
                    "pair": pair,
                    "arm": "defect",
                    "critic": name,
                    "kind": kind,
                    "verdict": "FIX",
                    "outcome": "hit" if pair in caught else "miss",
                }
            )
            rows.append(
                {
                    "pair": pair,
                    "arm": "control",
                    "critic": name,
                    "kind": kind,
                    "verdict": "PASS",
                    "outcome": "clear",
                }
            )
    result = evals.tally(rows, PAIRS, "docs")
    assert (result["critic_score"], result["baseline_score"]) == (5, 3)
    assert result["comparison"] == "value shown"
    assert sorted(result["per_reviewer"]) == ["baseline-docs", "docs-critic"]
    assert result["clean_pairs"]["pairs"] == []


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


def _panel_rows(kind: str, caught: set[str], alarms: set[str], pairs: list[str]) -> list[dict]:
    """One kind's three reviewers: the first catches `caught`, the last alarms on `alarms`."""
    names = evals.CRITICS if kind == "critic" else tuple(evals.BASELINES)
    rows = []
    for pair in pairs:
        for name in names:
            for arm, outcome in (
                ("defect", "hit" if pair in caught and name == names[0] else "miss"),
                ("control", "false alarm" if pair in alarms and name == names[-1] else "clear"),
            ):
                rows.append(
                    {
                        "pair": pair,
                        "arm": arm,
                        "critic": name,
                        "kind": kind,
                        "verdict": "REVISE",
                        "outcome": outcome,
                    }
                )
    return rows


@pytest.mark.parametrize(
    "baseline_caught, expected",
    [
        ({"a", "b", "c"}, "value shown"),  # 5 against 3
        ({"a", "b", "c", "d"}, "no measurable difference"),  # 5 against 4
        (set(PAIRS), "no measurable difference"),  # 5 against 5
    ],
)
def test_the_comparison_reads_the_margin(baseline_caught, expected):
    rows = _panel_rows("critic", set(PAIRS), set(), PAIRS)
    rows += _panel_rows("baseline", baseline_caught, set(), PAIRS)
    result = evals.tally(rows, PAIRS)
    assert result["comparison"] == expected
    assert (result["panel_hits"], result["baseline_hits"]) == (5, len(baseline_caught))


def test_the_comparison_can_find_the_critics_worse():
    rows = _panel_rows("critic", {"a", "b", "c", "d"}, {"a", "b"}, PAIRS)  # 4 - 2 = 2
    rows += _panel_rows("baseline", {"a", "b", "c", "d"}, set(), PAIRS)  # 4 - 0 = 4
    result = evals.tally(rows, PAIRS)
    assert (result["critic_score"], result["baseline_score"]) == (2, 4)
    assert result["comparison"] == "worse than baseline"


def test_false_alarms_count_against_a_panel():
    rows = _panel_rows("critic", set(PAIRS), {"a", "b", "c"}, PAIRS)  # 5 - 3 = 2
    rows += _panel_rows("baseline", set(PAIRS), set(), PAIRS)  # 5 - 0 = 5
    assert evals.tally(rows, PAIRS)["comparison"] == "worse than baseline"


def test_the_clean_pair_readout_counts_only_its_two_pairs():
    pairs = ["ftol-stall", "manifest-key", "obs-routing", "sorted-glob", "stale-affine"]
    rows = _panel_rows("critic", {"sorted-glob", "ftol-stall"}, {"stale-affine"}, pairs)
    rows += _panel_rows("baseline", {"ftol-stall"}, set(), pairs)
    clean = evals.tally(rows, pairs)["clean_pairs"]
    assert clean == {
        "pairs": ["sorted-glob", "stale-affine"],
        "critic caught": 1,
        "critic false alarms": 1,
        "baseline caught": 0,
        "baseline false alarms": 0,
    }


def test_a_run_without_baseline_rows_reads_only_the_first_bar():
    result = evals.tally(_rows(set(PAIRS), set(), PAIRS), PAIRS)
    assert "comparison" not in result and result["passes"] is True


def test_the_bar_passes_at_four_caught_and_one_false_alarm():
    result = evals.tally(_rows({"a", "b", "c", "d"}, {"e"}, PAIRS), PAIRS)
    assert (result["panel_hits"], result["panel_false_alarms"], result["passes"]) == (4, 1, True)
    assert result["per_reviewer"]["implementation-critic"]["hits"] == 4
    assert result["per_reviewer"]["red-team"]["false alarms"] == 1


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
    counts = evals.tally(_rows(set(), set(), PAIRS), PAIRS)["per_reviewer"]["method-critic"]
    assert counts["blocking verdicts, defect arm"] == 5
    assert counts["blocking verdicts, control arm"] == 0


# ------------------------------------------------------------ end to end


def _scored_run(tmp_path, monkeypatch, calibration_answer, baseline=False):
    """A run of two reviews, judged, with every calibration answered the same way.

    With ``baseline``, a third review by a baseline reviewer, which misses.
    """
    monkeypatch.setattr(evals, "RUNS", tmp_path)
    run_dir = tmp_path / "t"
    (run_dir / "responses").mkdir(parents=True)
    (run_dir / "judgements").mkdir()
    snapshot = "/tmp/pyvwf-checkouts/abcd1234"
    tasks = [
        {"token": "t1", "pair": "obs-routing", "arm": "defect", "critic": "implementation-critic"},
        {"token": "t2", "pair": "obs-routing", "arm": "control", "critic": "red-team"},
    ]
    if baseline:
        tasks.append(
            {
                "token": "t3",
                "pair": "obs-routing",
                "arm": "defect",
                "critic": "baseline-2",
                "kind": "baseline",
            }
        )
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
    if baseline:
        (run_dir / "responses" / "t3.md").write_text("**Verdict:** PROCEED\n\nLooks fine.\n")
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
    if baseline:
        (run_dir / "judgements" / "t3.md").write_text("FOUND: NO\nQUOTE: NONE")
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


def test_a_scored_run_with_a_baseline_reads_the_comparison(tmp_path, monkeypatch):
    result = _scored_run(tmp_path, monkeypatch, _honest, baseline=True)
    assert (result["panel_hits"], result["baseline_hits"]) == (1, 0)
    assert result["comparison"] == "no measurable difference"  # 1 against 0
    summary = (tmp_path / "t" / "results.md").read_text()
    assert "Comparison: critic score 1, baseline score 0: no measurable difference" in summary
    assert "implementation-critic against baseline-2: hits 1 and 0" in summary


def test_an_uncalibrated_judge_leaves_the_comparison_unread(tmp_path, monkeypatch):
    result = _scored_run(
        tmp_path, monkeypatch, lambda kind, text: _honest("positive", text), baseline=True
    )
    assert (result["passes"], result["baseline_passes"], result["comparison"]) == (None, None, None)
    assert "baseline score 0: not read" in (tmp_path / "t" / "results.md").read_text()


def test_a_scored_docs_run_excludes_a_command_outside_the_checkout(tmp_path, monkeypatch):
    """End to end on the docs suite: a hit, and a review excluded for its command."""
    monkeypatch.setattr(evals, "RUNS", tmp_path)
    run_dir = tmp_path / "d"
    (run_dir / "responses").mkdir(parents=True)
    (run_dir / "judgements").mkdir()
    tasks = [
        {
            "token": "d1",
            "pair": "file-count",
            "arm": "defect",
            "critic": "docs-critic",
            "kind": "critic",
        },
        {
            "token": "d2",
            "pair": "file-count",
            "arm": "defect",
            "critic": "baseline-docs",
            "kind": "baseline",
        },
    ]
    record = {
        "label": "d",
        "suite": "docs",
        "critic_model": None,
        "judge_model": "j",
        "harness_commit": "0" * 40,
        "snapshots": {"file-count": {"dir": SNAP}},
        "tasks": tasks,
    }
    (run_dir / "run.json").write_text(evals.json.dumps(record))
    (run_dir / "responses" / "d1.md").write_text(DOCS_REVIEW.format(extra="none"))
    (run_dir / "responses" / "d2.md").write_text(
        DOCS_REVIEW.format(extra="wc -l /Users/me/repo/CHANGELOG.md")
    )
    evals.judge(run_dir, "d")
    record = evals.json.loads((run_dir / "run.json").read_text())
    cases = {c["pair"]: c for c in evals.load_cases(DOCS["cases"], DOCS["critics"])}
    assert sorted({c["pair"] for c in record["calibration"]}) == sorted(cases)
    for cal in record["calibration"]:
        text = cases[cal["pair"]][f"judge_{cal['kind']}"]
        (run_dir / "judgements" / f"{cal['token']}.md").write_text(_honest(cal["kind"], text))
    for name in ("d1", "d2"):
        (run_dir / "judgements" / f"{name}.md").write_text(
            "FOUND: YES\nQUOTE: the count does not recount"
        )
    result = evals.score(run_dir)
    outcomes = {r["critic"]: r["outcome"] for r in result["rows"]}
    assert outcomes == {"docs-critic": "hit", "baseline-docs": "excluded"}
    assert result["judge_calibrated"] is True
    table = (run_dir / "results.md").read_text()
    assert "docs-critic against baseline-docs" in table and "Clean pairs" not in table
