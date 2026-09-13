"""The curve library study's driver: one condition of one row, per process.

Registered in ``docs/findings/method-curve-library-prereg.md``. A condition
changes the curve library a row runs on, or the model key each unit is
assigned, or both, and holds everything else at the row's scorecard
configuration.

**Why the override check refuses rather than reports.** Two of this study's
conditions are reassignments of model keys, and two of its registered
predictions say the reassignment will change nothing measurable (P2 and P4).
So a condition whose overrides were never applied produces exactly the result
those predictions expect: the run simulates the row as it already was, the
paired interval covers zero, and the gate records a null. **The study's most
likely bug and its most likely true result are indistinguishable from the
outside.** The only way to tell them apart is to refuse to run a condition on a
fleet that is not the one it asked for, before the fit begins. A check that
only recorded the discrepancy would be read after the result, if at all, and
the result would look fine.

The same reasoning as the ERA5 extent guard: a run on input nobody verified is
not evidence, however clean its metrics look.

What is checked, before train and again before evaluate:

- every unit the condition asked to reassign is present in the fleet the run
  will fit, and carries the key it asked for;
- no other unit's key moved;
- the run's curve library is the one the condition names, by sha256.

Any failure raises ``OverrideError`` with the count, the capacity share and the
first few units, and no run directory is written.

The requested overrides are also written to the run directory as
``curve_overrides.csv``, for provenance. The file is the record; the refusal is
the mechanism.

Usage, from the repository root, one region per process:

    PYVWF_INPUT=<root> PYTHONPATH=src:scripts/analysis python \\
        scripts/analysis/curve_library_study.py <CODE> <condition> <out_dir>
"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vwf.harness import driver  # noqa: E402
from vwf.harness.regions import load_region  # noqa: E402

#: How many offending units an error message names. Enough to recognise the
#: pattern, few enough to read.
SHOWN = 5


class OverrideError(RuntimeError):
    """A condition's fleet is not the one it asked for."""


def check_overrides(turb_info: pd.DataFrame, overrides: pd.Series, phase: str) -> None:
    """Refuse unless the fleet carries exactly the keys the condition asked for.

    Args:
        turb_info: the fleet the run will fit, with ``ID``, ``model`` and
            ``capacity``.
        overrides: requested key per unit ID. An ID absent from the fleet is a
            failure, not a silent skip: it usually means the two sides key on
            different types, which is how an override silently does nothing.
        phase: ``train`` or ``evaluate``, named in the error.
    """
    fleet = turb_info.assign(ID=turb_info["ID"].astype(str))
    wanted = overrides.copy()
    wanted.index = wanted.index.astype(str)
    capacity = pd.to_numeric(fleet["capacity"], errors="coerce").fillna(0.0)
    total = float(capacity.sum())

    missing = sorted(set(wanted.index) - set(fleet["ID"]))
    if missing:
        raise OverrideError(
            f"{phase}: {len(missing)} of {len(wanted)} requested units are not in the "
            f"fleet, so their override could not apply; for example {missing[:SHOWN]}. "
            "An override that reaches no unit produces the same result as a condition "
            "that changes nothing, which is what this study's predictions expect."
        )

    applied = fleet.set_index("ID")["model"].astype(str)
    asked = wanted.astype(str)
    wrong = asked.index[applied.loc[asked.index].to_numpy() != asked.to_numpy()]
    if len(wrong):
        share = float(capacity[fleet["ID"].isin(wrong)].sum()) / total if total else 0.0
        examples = [f"{u}: {applied[u]!r} not {asked[u]!r}" for u in list(wrong)[:SHOWN]]
        raise OverrideError(
            f"{phase}: {len(wrong)} units carry a key the condition did not ask for, "
            f"{share:.2%} of fleet capacity; for example {examples}."
        )


def check_library(run_dir: Path, expected_sha256: str | None) -> None:
    """Refuse unless the run resolved the curve library the condition names."""
    if not expected_sha256:
        return
    import json
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    got = manifest["curve_library"]["power_curves_sha256"]
    if got != expected_sha256:
        raise OverrideError(
            f"{run_dir.name}: curve library sha256 {got[:12]} is not the "
            f"{expected_sha256[:12]} this condition names. The run used a different "
            "library from the one its result would be attributed to."
        )


def apply_overrides(turb_info: pd.DataFrame, overrides: pd.Series) -> pd.DataFrame:
    """The fleet with the condition's keys in place of its own."""
    out = turb_info.copy()
    ids = out["ID"].astype(str)
    wanted = overrides.copy()
    wanted.index = wanted.index.astype(str)
    out["model"] = ids.map(wanted).fillna(out["model"])
    return out


def patched_fleet(overrides: pd.Series, phase: str):
    """A decorator for ``train_set`` or ``val_set`` that overrides and checks.

    The override is applied to the frame the run will actually fit, and checked
    on the same frame, so nothing between the two can undo it.
    """
    def wrap(loader):
        def loaded(*args, **kwargs):
            obs, turb_info, reanalysis, curves = loader(*args, **kwargs)
            turb_info = apply_overrides(turb_info, overrides)
            check_overrides(turb_info, overrides, phase)
            return obs, turb_info, reanalysis, curves
        return loaded
    return wrap


def run_condition(code: str, condition: str, out_root: Path, overrides: pd.Series,
                  *, config: Path, library_sha256: str | None = None,
                  mode: str = "all") -> tuple[Path, Path]:
    """Train and evaluate one row under one condition, or refuse."""
    spec = load_region(config)
    out_root = Path(out_root)
    train_loader, val_loader = driver.train_set, driver.val_set
    driver.train_set = patched_fleet(overrides, "train")(train_loader)
    driver.val_set = patched_fleet(overrides, "evaluate")(val_loader)
    try:
        train_dir = driver.run_train(spec, out_root, mode=mode, run_name=condition)
        check_library(train_dir, library_sha256)
        _write_overrides(train_dir, overrides)
        eval_dir = driver.run_evaluate(spec, train_dir, out_root, mode=mode,
                                       run_name=condition)
        check_library(eval_dir, library_sha256)
        _write_overrides(eval_dir, overrides)
    finally:
        driver.train_set, driver.val_set = train_loader, val_loader
    return train_dir, eval_dir


def _write_overrides(run_dir: Path, overrides: pd.Series) -> None:
    overrides.rename("model").rename_axis("ID").to_frame().to_csv(
        run_dir / "curve_overrides.csv")


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    code, condition, out_dir, overrides_csv = sys.argv[1:]
    table = pd.read_csv(overrides_csv, dtype=str).set_index("ID")["model"]
    config = Path("configs/regions/scorecard") / f"{_stem(code)}.toml"
    train_dir, eval_dir = run_condition(
        code, condition, Path(out_dir), table, config=config,
        library_sha256=os.environ.get("PYVWF_EXPECT_CURVES_SHA256"))
    print(f"{code} {condition}: {train_dir}, {eval_dir}")


def _stem(code: str) -> str:
    import baseline_bootstrap as bb
    return bb.CONFIGS[code]


if __name__ == "__main__":
    main()
