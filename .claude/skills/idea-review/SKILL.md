---
name: idea-review
description: Take a research or engineering idea for PyVWF through a scout, a draft proposal, three independent critiques and a revision, then present the result and stop. Use when the maintainer invokes /idea-review with an idea.
argument-hint: "<the idea, in a sentence or a paragraph>"
disable-model-invocation: true
---

# idea-review

The idea: $ARGUMENTS

You orchestrate. Subagents cannot call each other, so you run every step and
pass text between them in their prompts. Write no files in the repository at
any step. Terms follow `docs/CONTEXT.md`. No em dashes anywhere.

## 1. Scout

Call `prior-work-scout` with the idea. Wait for its brief.

If the brief shows the idea was already tested and failed, say so, cite the
document, and ask whether to continue before doing anything else.

## 2. Draft

Write a proposal of at most 400 words with these headings:

- **Claim:** what would be true if the idea works.
- **Mechanism:** why it should work, physically or statistically.
- **Change:** what code or configuration changes, with paths.
- **Test:** rows, conditions, training years and test year, what is held fixed.
- **Gates:** pass and fail criteria fixed before any run.
- **Expected effect and size.**
- **Risks.**

## 3. Critique, in parallel

Call `method-critic`, `implementation-critic` and `red-team` at the same time.
Give each the idea, the scout brief and the proposal. Do not give any critic
another critic's output.

## 4. Revise

Collect the verdicts. For every blocker, either change the proposal or write
one sentence on why it does not apply. Do not drop a blocker silently.

If any critic said REJECT, or a blocker changed the Test or Gates, run one
second round: send the revised proposal and the list of changes to the critics
that raised blockers only. Never more than two rounds.

## 5. Present, then stop

Give the maintainer:

1. The revised proposal.
2. A verdict table: critic, round 1, round 2.
3. Blockers resolved, and how.
4. **Disagreements left open.** Where critics disagree with each other or with
   the proposal, state both sides. Do not resolve them yourself.
5. The cheapest next step.

Then stop. Do not start implementing, running, or writing a preregistration.
If the maintainer approves, the preregistration goes through the `findings-doc`
skill.
