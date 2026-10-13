#!/usr/bin/env python3
"""Post-confirmation common-mask state-retention decomposition.

No implicit feature matrices are regenerated and no QNN is retrained.

The analysis starts from the common all-strict mask used by
run_selector_confound_audit.py. For each alternative selector, that common mask
is split into cases where the selector returns the same state as Zero and cases
where it returns a different state. Classical readouts are refit on the fixed
cached feature matrices. The established four-seed QNN predictions are reused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_selector_confound_audit as confound

PLAN = "common_mask_state_decomposition_v1"
ROOT = confound.ROOT
DEFAULT_OUT = ROOT / "outputs" / PLAN

METHODS = confound.METHODS
CLASSICAL_READOUTS = confound.CLASSICAL_READOUTS
QNN_REPS = confound.QNN_REPS
MASKS = confound.MASKS
FEATURE_ROOT = confound.FEATURE_ROOT

EXPECTED_POOLED = {
    "selected_mrbi": {"common": 674, "same": 491, "different": 183},
    "hybrid_ls": {"common": 674, "same": 482, "different": 192},
    "hybrid_random": {"common": 674, "same": 461, "different": 213},
    "multistart5": {"common": 674, "same": 163, "different": 511},
}


def aggregate_task_equal(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (state_subset, readout, method), g in rows.groupby(
        ["state_subset", "readout", "method"], sort=False
    ):
        if readout == "qnn":
            seed_level = (
                g.groupby(["dataset", "seed"], as_index=False)
                .agg(
                    delta=("accuracy_delta", "mean"),
                    disagreement=("prediction_disagreement", "mean"),
                    n=("n", "first"),
                )
            )
        else:
            seed_level = g.rename(
                columns={
                    "accuracy_delta": "delta",
                    "prediction_disagreement": "disagreement",
                }
            )[["dataset", "seed", "delta", "disagreement", "n"]]

        nonempty = seed_level[np.isfinite(seed_level["delta"])].copy()
        if nonempty.empty:
            continue

        task_level = (
            nonempty.groupby("dataset", as_index=False)
            .agg(
                delta=("delta", "mean"),
                disagreement=("disagreement", "mean"),
                mean_n=("n", "mean"),
            )
        )
        vals = task_level["delta"].to_numpy(float)
        tol = 1e-12
        out.append(
            {
                "state_subset": state_subset,
                "readout": readout,
                "method": method,
                "task_equal_accuracy_delta": float(np.mean(vals)),
                "task_equal_prediction_disagreement": float(
                    task_level["disagreement"].mean()
                ),
                "positive_tasks": int(np.sum(vals > tol)),
                "neutral_tasks": int(np.sum(np.abs(vals) <= tol)),
                "negative_tasks": int(np.sum(vals < -tol)),
                "n_tasks": int(len(task_level)),
                "mean_mask_size_per_nonempty_task_seed": float(
                    task_level["mean_n"].mean()
                ),
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if not args.run:
        raise SystemExit("Use --run")

    if not MASKS.exists():
        raise SystemExit(
            f"Missing masks: {MASKS}\n"
            "Run scripts/run_selection_quality_both_strict_audit.py first."
        )

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    masks = pd.read_csv(MASKS)
    lock = confound.confirm.load_lock()
    datasets = list(lock["development_evidence"]["datasets"])
    seeds = list(lock["confirmation_plan"]["seeds"])

    count_rows = []
    result_rows = []

    for dataset in datasets:
        for seed in seeds:
            print(f"STATE_DECOMP_START dataset={dataset} seed={seed}", flush=True)

            feature_path = FEATURE_ROOT / f"seed{seed}" / f"{dataset}.npz"
            if not feature_path.exists():
                raise SystemExit(f"Missing feature cache: {feature_path}")

            ytr, yte, reps = confound.allpaths.load_feature_cache(feature_path)
            ytr = np.asarray(ytr, dtype=int)
            yte = np.asarray(yte, dtype=int)

            mg = masks[
                (masks["dataset"] == dataset)
                & (masks["seed"].astype(int) == int(seed))
            ].sort_values("position").reset_index(drop=True)

            if len(mg) != len(yte):
                raise RuntimeError(
                    f"Mask length mismatch {dataset} seed={seed}: "
                    f"{len(mg)} != {len(yte)}"
                )
            if not np.array_equal(
                mg["position"].to_numpy(int), np.arange(len(yte))
            ):
                raise RuntimeError(f"Mask position mismatch {dataset} seed={seed}")

            required = {"zero_strict"}
            for method in METHODS:
                required |= {f"{method}_strict", f"{method}_different_state"}
            missing = sorted(required - set(mg.columns))
            if missing:
                raise RuntimeError(f"Mask columns missing: {missing}")

            all_strict = mg["zero_strict"].to_numpy(int).astype(bool)
            any_changed = np.zeros(len(mg), dtype=bool)
            for method in METHODS:
                all_strict &= mg[f"{method}_strict"].to_numpy(int).astype(bool)
                any_changed |= mg[f"{method}_different_state"].to_numpy(int).astype(bool)
            common_mask = all_strict & any_changed

            classical_pred = {}
            for readout in CLASSICAL_READOUTS:
                classical_pred[readout] = {}
                for method in ("zero",) + METHODS:
                    Xtr, Xte = reps[method]
                    model = confound.classical.make_model(readout, seed)
                    model.fit(Xtr, ytr)
                    classical_pred[readout][method] = np.asarray(
                        model.predict(Xte), dtype=int
                    )

            for method in METHODS:
                changed = mg[f"{method}_different_state"].to_numpy(int).astype(bool)
                same_mask = common_mask & ~changed
                diff_mask = common_mask & changed

                count_rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "method": method,
                        "common_n": int(np.sum(common_mask)),
                        "same_n": int(np.sum(same_mask)),
                        "different_n": int(np.sum(diff_mask)),
                    }
                )

                for readout in CLASSICAL_READOUTS:
                    p0 = classical_pred[readout]["zero"]
                    pm = classical_pred[readout][method]
                    for state_subset, mask in (
                        ("same_state", same_mask),
                        ("different_state", diff_mask),
                    ):
                        delta, disagree, n = confound.sample_accuracy_delta(
                            yte, p0, pm, mask
                        )
                        result_rows.append(
                            {
                                "dataset": dataset,
                                "seed": int(seed),
                                "qnn_rep": -1,
                                "readout": readout,
                                "method": method,
                                "state_subset": state_subset,
                                "n": n,
                                "accuracy_delta": delta,
                                "prediction_disagreement": disagree,
                            }
                        )

                for rep in range(QNN_REPS):
                    p0 = confound.load_qnn_pred(dataset, seed, rep, "zero")
                    pm = confound.load_qnn_pred(dataset, seed, rep, method)
                    for state_subset, mask in (
                        ("same_state", same_mask),
                        ("different_state", diff_mask),
                    ):
                        delta, disagree, n = confound.sample_accuracy_delta(
                            yte, p0, pm, mask
                        )
                        result_rows.append(
                            {
                                "dataset": dataset,
                                "seed": int(seed),
                                "qnn_rep": int(rep),
                                "readout": "qnn",
                                "method": method,
                                "state_subset": state_subset,
                                "n": n,
                                "accuracy_delta": delta,
                                "prediction_disagreement": disagree,
                            }
                        )

            print(
                f"STATE_DECOMP_DONE dataset={dataset} seed={seed} "
                f"common_n={int(np.sum(common_mask))}",
                flush=True,
            )

    counts = pd.DataFrame(count_rows)
    rows = pd.DataFrame(result_rows)
    overall = aggregate_task_equal(rows)

    pooled = (
        counts.groupby("method", as_index=False)[["common_n", "same_n", "different_n"]]
        .sum()
        .rename(
            columns={
                "common_n": "common_n_pooled",
                "same_n": "same_n_pooled",
                "different_n": "different_n_pooled",
            }
        )
    )

    for method, expected in EXPECTED_POOLED.items():
        row = pooled[pooled["method"] == method]
        if len(row) != 1:
            raise AssertionError(f"Missing pooled count row for {method}")
        row = row.iloc[0]
        actual = {
            "common": int(row["common_n_pooled"]),
            "same": int(row["same_n_pooled"]),
            "different": int(row["different_n_pooled"]),
        }
        if actual != expected:
            raise AssertionError(
                f"Pooled-count mismatch for {method}: {actual} != {expected}"
            )

    compact = overall[
        [
            "state_subset",
            "readout",
            "method",
            "task_equal_accuracy_delta",
            "positive_tasks",
            "neutral_tasks",
            "negative_tasks",
            "n_tasks",
        ]
    ].copy()

    counts.to_csv(out / "dataset_seed_counts.csv", index=False)
    pooled.to_csv(out / "pooled_counts.csv", index=False)
    rows.to_csv(out / "dataset_seed_readout_summary.csv", index=False)
    overall.to_csv(out / "overall_summary.csv", index=False)
    compact.to_csv(out / "compact_summary.csv", index=False)

    summary = {
        "plan": PLAN,
        "analysis_role": (
            "post-confirmation descriptive common-mask state-retention decomposition"
        ),
        "uses_new_feature_generation": False,
        "uses_new_qnn_training": False,
        "common_mask": (
            "Zero and all four alternative selectors are strict on the same sample, "
            "and at least one alternative returned state differs from Zero."
        ),
        "state_subsets": {
            "same_state": (
                "Common-mask cases where the compared selector returns the same "
                "state as Zero under the established state-agreement threshold."
            ),
            "different_state": (
                "Complementary common-mask cases where the compared selector "
                "returns a different state from Zero."
            ),
        },
        "interpretation_limit": (
            "The different-state contrast is not a causal test-state intervention "
            "because each selector-specific readout is fitted on its own training "
            "representation."
        ),
        "expected_pooled_counts": EXPECTED_POOLED,
        "files": [
            "dataset_seed_counts.csv",
            "pooled_counts.csv",
            "dataset_seed_readout_summary.csv",
            "overall_summary.csv",
            "compact_summary.csv",
        ],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("COMMON_MASK_STATE_DECOMPOSITION_OK", flush=True)


if __name__ == "__main__":
    main()
