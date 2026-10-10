#!/usr/bin/env python3
"""
QNN both-strict/different-state selection-quality audit.

Scientific question
-------------------
When Zero and an alternative numerical selector BOTH recover strict roots but
return DIFFERENT implicit states, what happens to the downstream QNN?

This audit is deliberately downstream-only:
- it reuses the already fixed/cached implicit feature matrices,
- it reuses the already computed method-specific both-strict/different masks,
- it uses the established four QNN readout seeds,
- it does NOT rerun MRBI/LS/Random/Multistart feature generation,
- it does NOT change the implicit operator or selection policy.

Methods
-------
selected_mrbi, hybrid_ls, hybrid_random, multistart5 versus zero.

Primary descriptive quantities on each method-specific mask
------------------------------------------------------------
1) QNN sample-accuracy delta: method - Zero
2) QNN prediction disagreement
3) counts of:
   - method correct / Zero wrong
   - Zero correct / method wrong
4) subset balanced-accuracy delta when both classes are present

Aggregation
-----------
QNN reps are averaged within each dataset/data-seed case.
Then:
- pooled sample-level summaries are reported across all available masked cases
  and all four QNN reps;
- task-equal summaries average data seeds within each task, then tasks equally.

The analysis is post-confirmation and descriptive.

Expected inputs
---------------
outputs/selection_quality_both_strict_v1/case_masks.csv
outputs/qnn_multiseed_all_paths_v1/features/seed*/<dataset>.npz
outputs/qnn_multiseed_all_paths_v1/checkpoint.json   (optional reproduction check)

Outputs
-------
outputs/selection_quality_qnn_both_strict_v1/
  predictions/*.npz
  training_metrics.csv
  dataset_seed_rep_summary.csv
  dataset_summary.csv
  overall_summary.csv
  summary.json

The predictions/ directory and training_metrics.csv are written incrementally,
so the run can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

import main_qnn_benchmark as bench
import run_locked_selected_profile_confirmation as confirm
import run_qnn_multiseed_all_paths as allpaths

PLAN = "selection_quality_qnn_both_strict_v1"
DEFAULT_OUT = ROOT / "outputs" / PLAN
MASKS = ROOT / "outputs" / "selection_quality_both_strict_v1" / "case_masks.csv"
FEATURE_ROOT = ROOT / "outputs" / "qnn_multiseed_all_paths_v1" / "features"
OLD_QNN_CHECKPOINT = ROOT / "outputs" / "qnn_multiseed_all_paths_v1" / "checkpoint.json"

METHODS = ("selected_mrbi", "hybrid_ls", "hybrid_random", "multistart5")
ALL_METHODS = ("zero",) + METHODS
REPLICATES = 4
REPRO_TOL = 1e-10


def qnn_seed_for(data_seed: int, rep: int) -> int:
    return int(data_seed + 777 + rep * 10000)


def safe_ba(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    return float(balanced_accuracy_score(y, pred))


def train_qnn_with_predictions(X_train, y_train, X_test, y_test, cfg, seed):
    if not bench.HAS_QNN:
        raise SystemExit(f"QNN dependency unavailable: {bench.QNN_IMPORT_ERROR}")

    torch = bench.torch
    nn = bench.nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr = torch.tensor(np.asarray(X_train), dtype=torch.float32)
    ytr = torch.tensor(np.asarray(y_train), dtype=torch.long)
    Xte = torch.tensor(np.asarray(X_test), dtype=torch.float32)

    model = bench.TorchQNN(
        X_train.shape[1],
        cfg.n_qubits,
        cfg.qnn_layers,
        seed,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.qnn_lr)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    for _epoch in range(cfg.qnn_epochs):
        perm = rng.permutation(len(Xtr))
        for start in range(0, len(Xtr), cfg.qnn_batch_size):
            idx = perm[start:start + cfg.qnn_batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()

    train_time = time.perf_counter() - t0

    with torch.no_grad():
        probs = torch.softmax(model(Xte), dim=1).cpu().numpy()
        pred = np.argmax(probs, axis=1).astype(int)

    metrics = bench.compute_metrics(y_test, pred, probs[:, 1])
    metrics["train_time_sec"] = float(train_time)
    return metrics, pred, probs


def prediction_path(pred_dir, dataset, data_seed, rep, method):
    return pred_dir / f"{dataset}__seed{data_seed}__rep{rep}__{method}.npz"


def load_old_ba():
    if not OLD_QNN_CHECKPOINT.exists():
        return {}
    payload = json.loads(OLD_QNN_CHECKPOINT.read_text(encoding="utf-8"))
    out = {}
    for r in payload.get("rows", []):
        out[(r["dataset"], int(r["data_seed"]), r["method"], int(r["qnn_rep"]))] = float(r["balanced_accuracy"])
    return out


def load_training_metrics(path):
    if not path.exists():
        return pd.DataFrame(columns=[
            "dataset", "data_seed", "qnn_rep", "qnn_seed", "method",
            "balanced_accuracy", "accuracy", "train_time_sec",
            "reproduction_checked", "reproduction_passed"
        ])
    return pd.read_csv(path)


def write_training_metrics(df, path):
    df.sort_values(["dataset", "data_seed", "qnn_rep", "method"]).to_csv(path, index=False)


def ensure_prediction(
    dataset, data_seed, rep, method, ytr, yte, reps, cfg,
    pred_dir, metrics_df, metrics_path, old_ba
):
    p = prediction_path(pred_dir, dataset, data_seed, rep, method)
    key = (dataset, int(data_seed), method, int(rep))
    qseed = qnn_seed_for(data_seed, rep)

    if p.exists():
        with np.load(p, allow_pickle=False) as raw:
            pred = np.asarray(raw["pred"], dtype=int)
            probs = np.asarray(raw["probs"], dtype=float)
            ba = float(raw["balanced_accuracy"])
        return pred, probs, ba, metrics_df

    Xtr, Xte = reps[method]
    print(
        f"QNN_START dataset={dataset} data_seed={data_seed} "
        f"rep={rep} method={method} qnn_seed={qseed}",
        flush=True,
    )

    m, pred, probs = train_qnn_with_predictions(Xtr, ytr, Xte, yte, cfg, qseed)
    ba = float(m["balanced_accuracy"])

    reproduction_checked = int(key in old_ba)
    reproduction_passed = 1
    if key in old_ba:
        expected = old_ba[key]
        reproduction_passed = int(np.isclose(ba, expected, atol=REPRO_TOL, rtol=0.0))
        if not reproduction_passed:
            raise RuntimeError(
                "Canonical QNN reproduction mismatch: "
                f"{key}, observed={ba:.12f}, expected={expected:.12f}"
            )

    np.savez_compressed(
        p,
        pred=pred.astype(np.int16),
        probs=probs.astype(np.float64),
        balanced_accuracy=np.asarray(ba),
        qnn_seed=np.asarray(qseed),
    )

    new_row = pd.DataFrame([{
        "dataset": dataset,
        "data_seed": int(data_seed),
        "qnn_rep": int(rep),
        "qnn_seed": int(qseed),
        "method": method,
        "balanced_accuracy": ba,
        "accuracy": float(m["accuracy"]),
        "train_time_sec": float(m["train_time_sec"]),
        "reproduction_checked": reproduction_checked,
        "reproduction_passed": reproduction_passed,
    }])

    if len(metrics_df):
        keep = ~(
            (metrics_df["dataset"] == dataset)
            & (metrics_df["data_seed"].astype(int) == int(data_seed))
            & (metrics_df["qnn_rep"].astype(int) == int(rep))
            & (metrics_df["method"] == method)
        )
        metrics_df = metrics_df.loc[keep].copy()

    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
    write_training_metrics(metrics_df, metrics_path)

    print(
        f"QNN_DONE dataset={dataset} data_seed={data_seed} "
        f"rep={rep} method={method} ba={ba:.4f}",
        flush=True,
    )
    return pred, probs, ba, metrics_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not MASKS.exists():
        raise SystemExit(
            f"Missing mask file: {MASKS}\n"
            "Run run_selection_quality_both_strict_audit.py first."
        )
    if not bench.HAS_QNN:
        raise SystemExit(f"QNN dependency unavailable: {bench.QNN_IMPORT_ERROR}")

    out = args.out_dir.resolve()
    pred_dir = out / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "training_metrics.csv"

    masks = pd.read_csv(MASKS)
    old_ba = load_old_ba()
    metrics_df = load_training_metrics(metrics_path)

    lock = confirm.load_lock()
    datasets = list(lock["development_evidence"]["datasets"])
    data_seeds = list(lock["confirmation_plan"]["seeds"])

    subset_rows = []

    for data_seed in data_seeds:
        for dataset in datasets:
            cache_path = FEATURE_ROOT / f"seed{data_seed}" / f"{dataset}.npz"
            if not cache_path.exists():
                raise SystemExit(f"Missing feature cache: {cache_path}")

            ytr, yte, reps = allpaths.load_feature_cache(cache_path)
            ytr = np.asarray(ytr, dtype=int)
            yte = np.asarray(yte, dtype=int)

            mg = masks[
                (masks["dataset"] == dataset)
                & (masks["seed"].astype(int) == int(data_seed))
            ].sort_values("position").reset_index(drop=True)

            if len(mg) != len(yte):
                raise RuntimeError(
                    f"Mask length mismatch {dataset} seed={data_seed}: "
                    f"{len(mg)} versus {len(yte)}"
                )
            if not np.array_equal(mg["position"].to_numpy(int), np.arange(len(yte))):
                raise RuntimeError(f"Mask position mismatch {dataset} seed={data_seed}")

            cfg = confirm.make_cfg(data_seed, lock)

            for rep in range(REPLICATES):
                pred = {}
                probs = {}
                full_ba = {}

                for method in ALL_METHODS:
                    pred[method], probs[method], full_ba[method], metrics_df = ensure_prediction(
                        dataset, data_seed, rep, method,
                        ytr, yte, reps, cfg,
                        pred_dir, metrics_df, metrics_path, old_ba
                    )

                for method in METHODS:
                    mask_col = f"{method}_both_strict_different"
                    if mask_col not in mg.columns:
                        raise RuntimeError(f"Missing {mask_col}")

                    mask = mg[mask_col].to_numpy(int).astype(bool)
                    n = int(mask.sum())

                    if n == 0:
                        row = {
                            "dataset": dataset,
                            "data_seed": int(data_seed),
                            "qnn_rep": int(rep),
                            "qnn_seed": qnn_seed_for(data_seed, rep),
                            "method": method,
                            "n_both_strict_different": 0,
                            "zero_accuracy_on_subset": np.nan,
                            "method_accuracy_on_subset": np.nan,
                            "accuracy_delta_method_minus_zero": np.nan,
                            "prediction_disagreement": np.nan,
                            "method_correct_zero_wrong": 0,
                            "zero_correct_method_wrong": 0,
                            "both_correct": 0,
                            "both_wrong": 0,
                            "zero_balanced_accuracy_on_subset": np.nan,
                            "method_balanced_accuracy_on_subset": np.nan,
                            "balanced_accuracy_delta_method_minus_zero": np.nan,
                            "mean_abs_probability_difference": np.nan,
                            "full_test_ba_zero": full_ba["zero"],
                            "full_test_ba_method": full_ba[method],
                        }
                    else:
                        yz = yte[mask]
                        p0 = pred["zero"][mask]
                        pm = pred[method][mask]
                        c0 = p0 == yz
                        cm = pm == yz

                        ba0 = safe_ba(yz, p0)
                        bam = safe_ba(yz, pm)
                        ba_delta = float(bam - ba0) if np.isfinite(ba0) and np.isfinite(bam) else np.nan

                        row = {
                            "dataset": dataset,
                            "data_seed": int(data_seed),
                            "qnn_rep": int(rep),
                            "qnn_seed": qnn_seed_for(data_seed, rep),
                            "method": method,
                            "n_both_strict_different": n,
                            "zero_accuracy_on_subset": float(np.mean(c0)),
                            "method_accuracy_on_subset": float(np.mean(cm)),
                            "accuracy_delta_method_minus_zero": float(np.mean(cm.astype(int) - c0.astype(int))),
                            "prediction_disagreement": float(np.mean(pm != p0)),
                            "method_correct_zero_wrong": int(np.sum(cm & ~c0)),
                            "zero_correct_method_wrong": int(np.sum(c0 & ~cm)),
                            "both_correct": int(np.sum(c0 & cm)),
                            "both_wrong": int(np.sum(~c0 & ~cm)),
                            "zero_balanced_accuracy_on_subset": ba0,
                            "method_balanced_accuracy_on_subset": bam,
                            "balanced_accuracy_delta_method_minus_zero": ba_delta,
                            "mean_abs_probability_difference": float(np.mean(np.abs(probs[method][mask, 1] - probs["zero"][mask, 1]))),
                            "full_test_ba_zero": full_ba["zero"],
                            "full_test_ba_method": full_ba[method],
                        }

                    subset_rows.append(row)

    subset_df = pd.DataFrame(subset_rows)
    subset_df.to_csv(out / "dataset_seed_rep_summary.csv", index=False)

    ds_seed_rows = []
    for (dataset, data_seed, method), g in subset_df.groupby(["dataset", "data_seed", "method"], sort=True):
        valid = g[np.isfinite(g["accuracy_delta_method_minus_zero"])].copy()
        if valid.empty:
            continue
        ds_seed_rows.append({
            "dataset": dataset,
            "data_seed": int(data_seed),
            "method": method,
            "n_both_strict_different": int(valid["n_both_strict_different"].iloc[0]),
            "mean_accuracy_delta_method_minus_zero": float(valid["accuracy_delta_method_minus_zero"].mean()),
            "mean_prediction_disagreement": float(valid["prediction_disagreement"].mean()),
            "mean_ba_delta_method_minus_zero": float(valid["balanced_accuracy_delta_method_minus_zero"].dropna().mean()) if valid["balanced_accuracy_delta_method_minus_zero"].notna().any() else np.nan,
            "qnn_rep_sd_accuracy_delta": float(valid["accuracy_delta_method_minus_zero"].std(ddof=1)) if len(valid) > 1 else np.nan,
            "method_correct_zero_wrong_sum": int(valid["method_correct_zero_wrong"].sum()),
            "zero_correct_method_wrong_sum": int(valid["zero_correct_method_wrong"].sum()),
        })

    ds_seed_df = pd.DataFrame(ds_seed_rows)
    ds_seed_df.to_csv(out / "dataset_seed_summary.csv", index=False)

    dataset_rows = []
    for (dataset, method), g in ds_seed_df.groupby(["dataset", "method"], sort=True):
        dataset_rows.append({
            "dataset": dataset,
            "method": method,
            "data_seed_cases_present": int(len(g)),
            "n_subset_samples_sum": int(g["n_both_strict_different"].sum()),
            "mean_qnn_accuracy_delta_method_minus_zero": float(g["mean_accuracy_delta_method_minus_zero"].mean()),
            "mean_qnn_prediction_disagreement": float(g["mean_prediction_disagreement"].mean()),
            "mean_qnn_ba_delta_method_minus_zero": float(g["mean_ba_delta_method_minus_zero"].dropna().mean()) if g["mean_ba_delta_method_minus_zero"].notna().any() else np.nan,
        })

    dataset_df = pd.DataFrame(dataset_rows)
    dataset_df.to_csv(out / "dataset_summary.csv", index=False)

    overall_rows = []
    summary = {
        "plan": PLAN,
        "analysis_role": "post-confirmation QNN selection-quality diagnostic",
        "qnn_replicates": REPLICATES,
        "qnn_seed_rule": "data_seed + 777 + rep * 10000",
        "state_subset": "method-specific Zero strict AND method strict AND different state",
        "results": {},
        "notes": [
            "Implicit feature matrices are reused from the established cache.",
            "The method-specific masks are reused from selection_quality_both_strict_v1.",
            "QNN training is repeated only because prior outputs did not store per-sample predictions.",
            "The four QNN seeds exactly match the established exploratory multiseed protocol.",
            "Existing full-test balanced accuracies are checked against the old QNN checkpoint when available.",
            "Cross-method subset sizes differ, so method-vs-method causal comparisons are not claimed.",
            "Results are descriptive and post-confirmation."
        ],
    }

    for method in METHODS:
        dset = dataset_df[dataset_df["method"] == method].copy()
        vals = dset["mean_qnn_accuracy_delta_method_minus_zero"].to_numpy(float)
        dis = dset["mean_qnn_prediction_disagreement"].to_numpy(float)

        raw = subset_df[
            (subset_df["method"] == method)
            & np.isfinite(subset_df["accuracy_delta_method_minus_zero"])
        ].copy()

        plus = int(raw["method_correct_zero_wrong"].sum())
        minus = int(raw["zero_correct_method_wrong"].sum())
        n_rep_samples = int(raw["n_both_strict_different"].sum())
        pooled_delta = float((plus - minus) / n_rep_samples) if n_rep_samples else None
        pooled_dis = None
        if n_rep_samples:
            pooled_dis = float(
                np.sum(raw["prediction_disagreement"].to_numpy(float) * raw["n_both_strict_different"].to_numpy(int)) / n_rep_samples
            )

        pn = {
            "positive": int(np.sum(vals > 1e-12)),
            "neutral": int(np.sum(np.abs(vals) <= 1e-12)),
            "negative": int(np.sum(vals < -1e-12)),
        }

        entry = {
            "datasets_present": int(len(dset)),
            "dataset_equal_mean_qnn_accuracy_delta": float(np.mean(vals)) if len(vals) else None,
            "dataset_equal_median_qnn_accuracy_delta": float(np.median(vals)) if len(vals) else None,
            "dataset_equal_mean_qnn_prediction_disagreement": float(np.mean(dis)) if len(dis) else None,
            "positive_neutral_negative_datasets": pn,
            "pooled_over_four_qnn_reps": {
                "replicated_subset_sample_count": n_rep_samples,
                "method_correct_zero_wrong": plus,
                "zero_correct_method_wrong": minus,
                "accuracy_delta_method_minus_zero": pooled_delta,
                "prediction_disagreement": pooled_dis,
            },
            "dataset_deltas": {
                r["dataset"]: float(r["mean_qnn_accuracy_delta_method_minus_zero"])
                for _, r in dset.iterrows()
            },
        }
        summary["results"][method] = entry

        overall_rows.append({
            "method": method,
            "datasets_present": entry["datasets_present"],
            "dataset_equal_mean_qnn_accuracy_delta": entry["dataset_equal_mean_qnn_accuracy_delta"],
            "dataset_equal_median_qnn_accuracy_delta": entry["dataset_equal_median_qnn_accuracy_delta"],
            "dataset_equal_mean_qnn_prediction_disagreement": entry["dataset_equal_mean_qnn_prediction_disagreement"],
            "positive_datasets": pn["positive"],
            "neutral_datasets": pn["neutral"],
            "negative_datasets": pn["negative"],
            "pooled_replicated_subset_sample_count": n_rep_samples,
            "pooled_method_correct_zero_wrong": plus,
            "pooled_zero_correct_method_wrong": minus,
            "pooled_accuracy_delta_method_minus_zero": pooled_delta,
            "pooled_prediction_disagreement": pooled_dis,
        })

    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(out / "overall_summary.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nQNN_BOTH_STRICT_SELECTION_QUALITY_OK")
    print(out / "summary.json")
    print("\nPRIMARY QNN SUMMARY")
    print(overall_df.to_string(index=False))


if __name__ == "__main__":
    main()
