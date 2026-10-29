Run 2026-10-29-docs: critics on their own models, judge claude-sonnet-5-5, harness at 929ebe4.

| Pair | Arm | Reviewer | Verdict | Judge | Quote in review | Files read | Outcome |
|---|---|---|---|---|---|---|---|
| cited-test | defect | docs-critic | FIX | YES | yes | inside | hit |
| cited-test | defect | baseline-docs | REVISE | YES | yes | inside | hit |
| cited-test | control | docs-critic | FIX | YES | yes | inside | false alarm |
| cited-test | control | baseline-docs | REVISE | NO | no | inside | clear |
| dropped-caveat | defect | docs-critic | FIX | NO | no | inside | miss |
| dropped-caveat | defect | baseline-docs | REVISE | NO | no | outside snapshot | excluded |
| dropped-caveat | control | docs-critic | FIX | YES | yes | inside | false alarm |
| dropped-caveat | control | baseline-docs | REVISE | NO | no | inside | clear |
| file-count | defect | docs-critic | FIX | YES | yes | inside | hit |
| file-count | defect | baseline-docs | REVISE | YES | yes | outside snapshot | excluded |
| file-count | control | docs-critic | FIX | NO | no | inside | clear |
| file-count | control | baseline-docs | REVISE | NO | no | inside | clear |
| pinned-halves | defect | docs-critic | FIX | YES | yes | inside | hit |
| pinned-halves | defect | baseline-docs | REVISE | NO | no | outside snapshot | excluded |
| pinned-halves | control | docs-critic | FIX | NO | no | outside snapshot | excluded |
| pinned-halves | control | baseline-docs | REVISE | NO | no | outside snapshot | excluded |
| reproduction-claim | defect | docs-critic | FIX | NO | no | inside | miss |
| reproduction-claim | defect | baseline-docs | REVISE | NO | no | inside | miss |
| reproduction-claim | control | docs-critic | FIX | NO | no | inside | clear |
| reproduction-claim | control | baseline-docs | REVISE | NO | no | outside snapshot | excluded |

| Reviewer | Hits | Noted | Misses | False alarms | Clear | Excluded |
|---|---|---|---|---|---|---|
| docs-critic | 3 | 0 | 2 | 2 | 2 | 1 |
| baseline-docs | 1 | 0 | 1 | 0 | 3 | 5 |

Judge calibration: passed.
Panel: 3 of 5 defects caught, 2 of 5 controls with a false alarm.
Bar: not passed (at least 4 caught, at most 1 false alarm).
Baseline panel: 1 of 5 defects caught, 0 of 5 controls with a false alarm; absolute bar not passed.
Comparison: critic score 1, baseline score 1: no measurable difference (value shown needs the critics 2 or more ahead).
docs-critic against baseline-docs: hits 3 and 1, false alarms 2 and 0.
