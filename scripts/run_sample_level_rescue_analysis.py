"""Sample-level rescue, multiplicity, and input-drive analysis for MRBI Quantum.

Post-confirmation mechanistic diagnostic.

Uses:
- outputs/all_path_rho_selector_sweep_v1/sample_rows.csv
- outputs/qnn_multiseed_all_paths_v1/features/seed*/<dataset>.npz

No implicit features are recomputed.
No QNN is trained.

Questions:
1. When Zero fails the strict criterion but multistart-5 returns a strict solution,
   does that numerical rescue improve sample-level classification?
2. When multiple strict roots are directly discovered, do numerical selectors
   disagree more often in their downstream predictions?
3. Is input-drive strength ||U x + b|| associated with discovered strict-root
   multiplicity?

Outputs:
outputs/sample_level_rescue_analysis_v1/
    sample_predictions.csv
    dataset_readout_summary.csv
    input_drive_summary.csv
    summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_locked_selected_profile_confirmation as confirm
import run_qnn_multiseed_all_paths as allpaths

PLAN = "sample_level_rescue_analysis_v1"
DEFAULT_SELECTOR_DIR = ROOT / "outputs" / "all_path_rho_selector_sweep_v1"
DEFAULT_FEATURE_ROOT = ROOT / "outputs" / "qnn_multiseed_all_paths_v1" / "features"
DEFAULT_OUT = ROOT / "outputs" / PLAN

METHODS = (
    "zero",
    "selected_mrbi",
    "hybrid_ls",
    "hybrid_random",
    "multistart5",
)
READOUTS = ("logreg", "svm_rbf", "mlp", "rf", "gb")
TOL = 1e-12


def make_model(name: str, seed: int):
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=3000),
        )
    if name == "svm_rbf":
        return make_pipeline(
            StandardScaler(),
            SVC(C=1.0, gamma="scale", probability=False),
        )
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                learning_rate_init=1e-3,
                alpha=1e-4,
                max_iter=3000,
                random_state=seed,
            ),
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=seed,
        )
    if name == "gb":
        return GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=seed,
        )
    raise ValueError(name)


def read_selector_rows(path: Path, rho: float) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing selector sample rows: {path}")
    df = pd.read_csv(path)
    needed = {
        "dataset",
        "seed",
        "rho",
        "position",
        "label",
        "input_drive_norm",
        "multistart_distinct_strict_roots_1e6",
        "multistart_multiple_strict_roots_1e6",
        "zero_success_from_multistart_probe",
        "multistart_selected_success",
    }
    missing = sorted(needed - set(df.columns))
    if missing:
        raise SystemExit(f"selector sample_rows.csv missing columns: {missing}")
    df = df[np.isclose(df["rho"].astype(float), float(rho), atol=1e-12)].copy()
    if df.empty:
        raise SystemExit(f"No selector sample rows found for rho={rho}")
    return df.sort_values(["dataset", "seed", "position"]).reset_index(drop=True)


def verify_alignment(group: pd.DataFrame, yte: np.ndarray, dataset: str, seed: int):
    g = group.sort_values("position")
    pos = g["position"].to_numpy(dtype=int)
    if not np.array_equal(pos, np.arange(len(yte), dtype=int)):
        raise RuntimeError(
            f"Position mismatch for {dataset} seed={seed}: "
            f"expected 0..{len(yte)-1}"
        )
    labels = g["label"].to_numpy(dtype=int)
    if not np.array_equal(labels, np.asarray(yte, dtype=int)):
        raise RuntimeError(
            f"Label/order mismatch for {dataset} seed={seed}"
        )


def masked_acc(y, pred, mask):
    mask = np.asarray(mask, dtype=bool)
    if int(np.sum(mask)) == 0:
        return np.nan
    return float(np.mean(np.asarray(pred)[mask] == np.asarray(y)[mask]))


def masked_disagreement(a, b, mask):
    mask = np.asarray(mask, dtype=bool)
    if int(np.sum(mask)) == 0:
        return np.nan
    return float(np.mean(np.asarray(a)[mask] != np.asarray(b)[mask]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rho", type=float, default=2.0)
    ap.add_argument("--selector-dir", type=Path, default=DEFAULT_SELECTOR_DIR)
    ap.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    selector_path = args.selector_dir / "sample_rows.csv"
    selector = read_selector_rows(selector_path, args.rho)

    lock = confirm.load_lock()
    datasets = list(lock["development_evidence"]["datasets"])
    seeds = list(lock["confirmation_plan"]["seeds"])

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = []

    for dataset in datasets:
        for seed in seeds:
            g = selector[
                (selector["dataset"] == dataset)
                & (selector["seed"].astype(int) == int(seed))
            ].copy()
            if g.empty:
                raise RuntimeError(
                    f"Missing selector rows for {dataset} seed={seed}"
                )

            feature_path = args.feature_root / f"seed{seed}" / f"{dataset}.npz"
            if not feature_path.exists():
                raise SystemExit(f"Missing cached feature file: {feature_path}")

            ytr, yte, reps = allpaths.load_feature_cache(feature_path)
            ytr = np.asarray(ytr, dtype=int)
            yte = np.asarray(yte, dtype=int)
            verify_alignment(g, yte, dataset, seed)
            g = g.sort_values("position").reset_index(drop=True)

            zero_success = g["zero_success_from_multistart_probe"].to_numpy(dtype=int)
            ms_success = g["multistart_selected_success"].to_numpy(dtype=int)
            multiple = g["multistart_multiple_strict_roots_1e6"].to_numpy(dtype=int)
            distinct = g["multistart_distinct_strict_roots_1e6"].to_numpy(dtype=int)
            rescue = (zero_success == 0) & (ms_success == 1)

            for readout in READOUTS:
                preds = {}
                for method in METHODS:
                    Xtr, Xte = reps[method]
                    model = make_model(readout, seed)
                    model.fit(Xtr, ytr)
                    preds[method] = np.asarray(model.predict(Xte), dtype=int)

                for i in range(len(yte)):
                    row = {
                        "dataset": dataset,
                        "seed": int(seed),
                        "rho": float(args.rho),
                        "position": int(i),
                        "readout": readout,
                        "label": int(yte[i]),
                        "zero_success": int(zero_success[i]),
                        "multistart_selected_success": int(ms_success[i]),
                        "multistart_rescue": int(rescue[i]),
                        "multiple_strict_roots": int(multiple[i]),
                        "distinct_strict_roots": int(distinct[i]),
                        "input_drive_norm": float(g.loc[i, "input_drive_norm"]),
                    }
                    for method in METHODS:
                        row[f"pred_{method}"] = int(preds[method][i])
                        row[f"correct_{method}"] = int(preds[method][i] == yte[i])
                    sample_rows.append(row)

                print(
                    f"RESCUE_DONE dataset={dataset} seed={seed} readout={readout} "
                    f"n_rescue={int(np.sum(rescue))} n_multi={int(np.sum(multiple))}",
                    flush=True,
                )

    sdf = pd.DataFrame(sample_rows)
    sdf.to_csv(out_dir / "sample_predictions.csv", index=False)

    dataset_rows = []
    for readout in READOUTS:
        for dataset in datasets:
            d = sdf[
                (sdf["readout"] == readout)
                & (sdf["dataset"] == dataset)
            ].copy()

            rescue = d["multistart_rescue"].to_numpy(dtype=int) == 1
            multi = d["multiple_strict_roots"].to_numpy(dtype=int) == 1

            row = {
                "readout": readout,
                "dataset": dataset,
                "n_samples": int(len(d)),
                "n_rescue_samples": int(np.sum(rescue)),
                "n_multiple_root_samples": int(np.sum(multi)),
            }

            if np.sum(rescue) > 0:
                z = d.loc[rescue, "correct_zero"].to_numpy(dtype=int)
                m = d.loc[rescue, "correct_multistart5"].to_numpy(dtype=int)
                row["rescue_acc_zero"] = float(np.mean(z))
                row["rescue_acc_multistart5"] = float(np.mean(m))
                row["rescue_delta_multistart_minus_zero"] = float(np.mean(m - z))
                row["rescue_wrong_to_correct"] = int(np.sum((z == 0) & (m == 1)))
                row["rescue_correct_to_wrong"] = int(np.sum((z == 1) & (m == 0)))
            else:
                row["rescue_acc_zero"] = np.nan
                row["rescue_acc_multistart5"] = np.nan
                row["rescue_delta_multistart_minus_zero"] = np.nan
                row["rescue_wrong_to_correct"] = 0
                row["rescue_correct_to_wrong"] = 0

            for method in METHODS:
                row[f"multi_root_acc_{method}"] = masked_acc(
                    d["label"].to_numpy(),
                    d[f"pred_{method}"].to_numpy(),
                    multi,
                )

            for method in METHODS[1:]:
                row[f"multi_root_disagree_{method}_vs_zero"] = masked_disagreement(
                    d[f"pred_{method}"].to_numpy(),
                    d["pred_zero"].to_numpy(),
                    multi,
                )

            row["multi_root_disagree_mrbi_vs_ls"] = masked_disagreement(
                d["pred_selected_mrbi"].to_numpy(),
                d["pred_hybrid_ls"].to_numpy(),
                multi,
            )

            dataset_rows.append(row)

    ddf = pd.DataFrame(dataset_rows)
    ddf.to_csv(out_dir / "dataset_readout_summary.csv", index=False)

    all_selector = pd.read_csv(selector_path)
    drive_rows = []

    for rho in sorted(all_selector["rho"].astype(float).unique()):
        for dataset in datasets:
            g = all_selector[
                np.isclose(all_selector["rho"].astype(float), float(rho), atol=1e-12)
                & (all_selector["dataset"] == dataset)
            ].copy()

            x = g["input_drive_norm"].to_numpy(dtype=float)
            y = g["multistart_distinct_strict_roots_1e6"].to_numpy(dtype=float)

            if len(x) >= 3 and not np.all(y == y[0]):
                res = spearmanr(x, y)
                corr = float(res.statistic)
                pval = float(res.pvalue)
            else:
                corr = np.nan
                pval = np.nan

            med = float(np.median(x))
            low = x <= med
            high = x > med
            multi = y >= 2

            drive_rows.append({
                "rho": float(rho),
                "dataset": dataset,
                "n_samples": int(len(x)),
                "spearman_drive_vs_distinct_roots": corr,
                "spearman_p": pval,
                "median_drive": med,
                "multiple_root_fraction_low_drive": (
                    float(np.mean(multi[low])) if np.any(low) else np.nan
                ),
                "multiple_root_fraction_high_drive": (
                    float(np.mean(multi[high])) if np.any(high) else np.nan
                ),
                "high_minus_low_multiple_root_fraction": (
                    float(np.mean(multi[high]) - np.mean(multi[low]))
                    if np.any(low) and np.any(high) else np.nan
                ),
            })

    drive_df = pd.DataFrame(drive_rows)
    drive_df.to_csv(out_dir / "input_drive_summary.csv", index=False)

    summary = {
        "plan": PLAN,
        "analysis_role": "post-confirmation exploratory mechanistic diagnostic",
        "rho_for_prediction_analysis": float(args.rho),
        "readouts": {},
        "input_drive": {},
    }

    for readout in READOUTS:
        d = ddf[ddf["readout"] == readout].copy()
        vals = d["rescue_delta_multistart_minus_zero"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]

        entry = {
            "datasets_with_rescue_samples": int(len(vals)),
            "dataset_mean_rescue_delta_multistart_minus_zero": (
                float(np.mean(vals)) if len(vals) else None
            ),
            "dataset_median_rescue_delta_multistart_minus_zero": (
                float(np.median(vals)) if len(vals) else None
            ),
            "rescue_positive_neutral_negative": {
                "positive": int(np.sum(vals > TOL)),
                "neutral": int(np.sum(np.abs(vals) <= TOL)),
                "negative": int(np.sum(vals < -TOL)),
            },
            "multi_root_disagreement_vs_zero": {},
        }

        for method in METHODS[1:]:
            x = d[f"multi_root_disagree_{method}_vs_zero"].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            entry["multi_root_disagreement_vs_zero"][method] = (
                float(np.mean(x)) if len(x) else None
            )

        x = d["multi_root_disagree_mrbi_vs_ls"].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        entry["multi_root_disagreement_mrbi_vs_ls"] = (
            float(np.mean(x)) if len(x) else None
        )
        summary["readouts"][readout] = entry

    for rho in sorted(drive_df["rho"].unique()):
        d = drive_df[np.isclose(drive_df["rho"], rho)].copy()
        cors = d["spearman_drive_vs_distinct_roots"].to_numpy(dtype=float)
        cors = cors[np.isfinite(cors)]
        gaps = d["high_minus_low_multiple_root_fraction"].to_numpy(dtype=float)
        gaps = gaps[np.isfinite(gaps)]

        summary["input_drive"][f"{rho:.2f}"] = {
            "mean_dataset_spearman": float(np.mean(cors)) if len(cors) else None,
            "median_dataset_spearman": float(np.median(cors)) if len(cors) else None,
            "negative_correlations": int(np.sum(cors < 0)),
            "positive_correlations": int(np.sum(cors > 0)),
            "mean_high_minus_low_multiple_root_fraction": (
                float(np.mean(gaps)) if len(gaps) else None
            ),
        }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"SAMPLE_LEVEL_RESCUE_ANALYSIS_OK {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
