#!/usr/bin/env python3
"""Robustness checks for the post-confirmation local geometry diagnostic.

Uses existing raw-state, standardized-feature, and strict-mask caches. It adds
strict-only and changed-pair views, pair directions relative to Zero, and
dataset-bootstrap uncertainty without rerunning implicit solvers or readouts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_locked_selected_profile_confirmation as confirm

PLAN = "local_feature_regularity_robustness_v1"
RAW_ROOT = ROOT / "outputs" / "local_feature_regularity_raw_v1" / "raw_features"
STD_ROOT = ROOT / "outputs" / "qnn_multiseed_all_paths_v1" / "features"
MASK_FILE = ROOT / "outputs" / "selection_quality_both_strict_v1" / "case_masks.csv"
OUT = ROOT / "outputs" / PLAN

METHODS = ("selected_mrbi", "hybrid_ls", "hybrid_random", "multistart5")
KS = (1, 3, 5)
STATE_TOL = 1e-6
PAIR_TIE_ATOL = 1e-12
BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 1729
EPS = 1e-12


def drel(a, b):
    num = np.linalg.norm(a - b, axis=-1)
    den = 1.0 + np.linalg.norm(a, axis=-1) + np.linalg.norm(b, axis=-1)
    return num / den


def knn_pairs(X, k):
    X = np.asarray(X, dtype=float)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    neigh = np.argsort(D, axis=1)[:, :k]
    pairs = sorted({tuple(sorted((i, int(j)))) for i in range(len(X)) for j in neigh[i]})
    return np.asarray(pairs, dtype=int)


def pair_amplification(X, Z, pairs):
    i, j = pairs[:, 0], pairs[:, 1]
    dx = np.linalg.norm(X[i] - X[j], axis=1)
    dz = np.linalg.norm(Z[i] - Z[j], axis=1)
    return dz, dz / (dx + EPS)


def summarize(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"n_pairs": 0, "median": np.nan, "p90": np.nan, "p95": np.nan}
    return {
        "n_pairs": int(len(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def load_raw(dataset, seed):
    path = RAW_ROOT / f"seed{seed}" / f"{dataset}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def load_std(dataset, seed):
    path = STD_ROOT / f"seed{seed}" / f"{dataset}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def bootstrap_mean_ci(values, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def sign_counts(values, tol=1e-12):
    values = np.asarray(values, dtype=float)
    return {
        "lower": int(np.sum(values < -tol)),
        "equal": int(np.sum(np.abs(values) <= tol)),
        "higher": int(np.sum(values > tol)),
    }


def mask_group(mask_df, dataset, seed, n):
    if mask_df is None:
        return None
    g = mask_df[
        (mask_df["dataset"] == dataset)
        & (mask_df["seed"].astype(int) == int(seed))
    ].sort_values("position").reset_index(drop=True)
    if len(g) != n:
        raise RuntimeError(f"Mask length mismatch: {dataset} seed={seed}: {len(g)} != {n}")
    return g


def find_strict_col(columns, method):
    """Return an exact per-sample strict-status column when present."""
    candidates = [
        f"{method}_strict",
        f"{method}_strict_success",
        f"{method}_is_strict",
        f"strict_{method}",
    ]
    for c in candidates:
        if c in columns:
            return c
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if not args.run:
        raise SystemExit("Use --run")

    OUT.mkdir(parents=True, exist_ok=True)
    lock = confirm.load_lock()
    datasets = list(lock["development_evidence"]["datasets"])
    seeds = list(lock["confirmation_plan"]["seeds"])

    mask_df = pd.read_csv(MASK_FILE) if MASK_FILE.exists() else None
    case_rows = []
    pair_shift_rows = []

    for dataset in datasets:
        for seed in seeds:
            raw = load_raw(dataset, seed)
            std = load_std(dataset, seed)

            X = np.asarray(raw["Xte_pca"], dtype=float)
            Z0_raw = np.asarray(raw["Zte_zero"], dtype=float)
            n = len(X)
            mg = mask_group(mask_df, dataset, seed, n)

            for k in KS:
                pairs = knn_pairs(X, k)
                i, j = pairs[:, 0], pairs[:, 1]
                dz0_raw, amp0_raw = pair_amplification(X, Z0_raw, pairs)

                for method in METHODS:
                    Zm_raw = np.asarray(raw[f"Zte_{method}"], dtype=float)
                    dzm_raw, ampm_raw = pair_amplification(X, Zm_raw, pairs)

                    changed_sample = drel(Zm_raw, Z0_raw) > STATE_TOL
                    changed_pair = changed_sample[i] | changed_sample[j]

                    strict_pair = None
                    strict_changed_pair = None
                    if mg is not None:
                        zc = find_strict_col(mg.columns, "zero")
                        mc = find_strict_col(mg.columns, method)
                        if zc is not None and mc is not None:
                            zs = mg[zc].to_numpy(int).astype(bool)
                            ms = mg[mc].to_numpy(int).astype(bool)
                            strict_pair = zs[i] & zs[j] & ms[i] & ms[j]
                            strict_changed_pair = strict_pair & changed_pair
                        else:
                            col = f"{method}_both_strict_different"
                            if col in mg.columns:
                                bs = mg[col].to_numpy(int).astype(bool)
                                strict_changed_pair = bs[i] | bs[j]

                    Z0_std = np.asarray(std["Zte_zero"], dtype=float)
                    Zm_std = np.asarray(std[f"Zte_{method}"], dtype=float)
                    _, amp0_std = pair_amplification(X, Z0_std, pairs)
                    _, ampm_std = pair_amplification(X, Zm_std, pairs)

                    views = [
                        ("raw_all", np.ones(len(pairs), dtype=bool), ampm_raw),
                        ("raw_changed_pairs", changed_pair, ampm_raw),
                        ("standardized_all", np.ones(len(pairs), dtype=bool), ampm_std),
                        ("standardized_changed_pairs", changed_pair, ampm_std),
                    ]
                    if strict_pair is not None:
                        views.append(("raw_strict_only", strict_pair, ampm_raw))
                        views.append(("standardized_strict_only", strict_pair, ampm_std))
                    if strict_changed_pair is not None:
                        strict_view = (
                            "strict_changed_pairs"
                            if strict_pair is not None
                            else "both_strict_different_endpoint_fallback"
                        )
                        views.append((f"raw_{strict_view}", strict_changed_pair, ampm_raw))
                        views.append((f"standardized_{strict_view}", strict_changed_pair, ampm_std))

                    for view, mask, vals in views:
                        sm = summarize(vals[mask])
                        case_rows.append({
                            "dataset": dataset,
                            "seed": int(seed),
                            "k": int(k),
                            "method": method,
                            "view": view,
                            **sm,
                        })

                    diff = dzm_raw - dz0_raw
                    smaller = diff < -PAIR_TIE_ATOL
                    equal = np.abs(diff) <= PAIR_TIE_ATOL
                    larger = diff > PAIR_TIE_ATOL
                    pair_shift_rows.append({
                        "dataset": dataset,
                        "seed": int(seed),
                        "k": int(k),
                        "method": method,
                        "n_pairs": int(len(pairs)),
                        "fraction_smaller_than_zero": float(np.mean(smaller)),
                        "fraction_equal_to_zero": float(np.mean(equal)),
                        "fraction_larger_than_zero": float(np.mean(larger)),
                        "changed_sample_fraction": float(np.mean(changed_sample)),
                        "changed_pair_fraction": float(np.mean(changed_pair)),
                    })

    case_df = pd.DataFrame(case_rows)
    shift_df = pd.DataFrame(pair_shift_rows)
    case_df.to_csv(OUT / "case_summary.csv", index=False)
    shift_df.to_csv(OUT / "pair_direction_vs_zero.csv", index=False)

    ds = (
        case_df.groupby(["dataset", "k", "method", "view"], as_index=False)
        .agg(
            n_pairs=("n_pairs", "sum"),
            median=("median", "mean"),
            p90=("p90", "mean"),
            p95=("p95", "mean"),
        )
    )
    ds.to_csv(OUT / "dataset_summary.csv", index=False)

    shift_ds = (
        shift_df.groupby(["dataset", "k", "method"], as_index=False)
        .agg(
            fraction_smaller_than_zero=("fraction_smaller_than_zero", "mean"),
            fraction_equal_to_zero=("fraction_equal_to_zero", "mean"),
            fraction_larger_than_zero=("fraction_larger_than_zero", "mean"),
            changed_sample_fraction=("changed_sample_fraction", "mean"),
            changed_pair_fraction=("changed_pair_fraction", "mean"),
        )
    )
    shift_ds.to_csv(OUT / "pair_direction_dataset_summary.csv", index=False)

    overall = (
        ds.groupby(["k", "method", "view"], as_index=False)
        .agg(
            median=("median", "mean"),
            p90=("p90", "mean"),
            p95=("p95", "mean"),
            datasets=("dataset", "nunique"),
        )
    )
    overall.to_csv(OUT / "overall_summary.csv", index=False)

    comparisons = (
        ("selected_mrbi", "hybrid_random"),
        ("selected_mrbi", "multistart5"),
        ("hybrid_ls", "hybrid_random"),
        ("hybrid_ls", "multistart5"),
    )
    comp_rows = []
    for view in sorted(ds["view"].unique()):
        for k in KS:
            dsk = ds[(ds["view"] == view) & (ds["k"] == k)]
            for a, b in comparisons:
                aa = dsk[dsk["method"] == a].set_index("dataset")
                bb = dsk[dsk["method"] == b].set_index("dataset")
                common = sorted(set(aa.index) & set(bb.index))
                for metric in ("p90", "p95"):
                    diffs = np.asarray(
                        [float(aa.loc[d, metric] - bb.loc[d, metric]) for d in common],
                        dtype=float,
                    )
                    lo, hi = bootstrap_mean_ci(
                        diffs,
                        seed=BOOTSTRAP_SEED + k + (0 if metric == "p90" else 100),
                    )
                    signs = sign_counts(diffs)
                    comp_rows.append({
                        "view": view,
                        "k": int(k),
                        "metric": metric,
                        "method_a": a,
                        "method_b": b,
                        "mean_difference_a_minus_b": float(np.mean(diffs)),
                        "bootstrap95_low": lo,
                        "bootstrap95_high": hi,
                        "datasets_lower_equal_higher": f"{signs['lower']}/{signs['equal']}/{signs['higher']}",
                        "datasets_lower": signs["lower"],
                        "datasets_equal": signs["equal"],
                        "datasets_higher": signs["higher"],
                    })

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUT / "structured_vs_arbitrary_uncertainty.csv", index=False)

    summary = {
        "plan": PLAN,
        "analysis_role": "post-confirmation descriptive robustness diagnostic",
        "state_change_threshold": STATE_TOL,
        "pair_tie_atol": PAIR_TIE_ATOL,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_unit": "dataset",
        "nested_k_choices": list(KS),
        "raw_geometry": "raw unstandardized returned equilibrium states",
        "standardized_geometry": "fixed standardized feature cache used by QNN readouts",
        "strict_mask_source": (
            str(MASK_FILE.relative_to(ROOT)) if MASK_FILE.exists() else None
        ),
        "strict_mask_note": (
            "If per-sample Zero and selector strict-status columns are present, "
            "strict-only pairs require both endpoints to be strict for both paths. "
            "If those columns are absent, the script reports a narrower fallback "
            "based on the established method-specific both-strict/different mask "
            "and labels that fallback explicitly."
            if MASK_FILE.exists()
            else "No strict mask file was available."
        ),
        "outputs": [
            "case_summary.csv",
            "dataset_summary.csv",
            "overall_summary.csv",
            "pair_direction_vs_zero.csv",
            "pair_direction_dataset_summary.csv",
            "structured_vs_arbitrary_uncertainty.csv",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("LOCAL_FEATURE_REGULARITY_ROBUSTNESS_OK")
    print(OUT)
    print("\nRAW ALL-PAIR STRUCTURED VS ARBITRARY")
    show = comp_df[comp_df["view"] == "raw_all"]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
