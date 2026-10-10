#!/usr/bin/env python3
"""Compute local geometry of raw solver-selected equilibrium features.

A shared k-nearest-neighbor graph is built in bounded PCA input space and reused
for all selectors. The script reports median, 90th-, and 95th-percentile local
amplification and caches raw returned test states for later robustness checks.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from itertools import combinations

import numpy as np

ROOT = Path(__file__).resolve().parent
if not (ROOT / "experiments").exists():
    ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

import main_qnn_benchmark as bench
import run_locked_selected_profile_confirmation as confirm
import run_hybrid_random_control as random_control
import run_least_squares_control as ls_control

PLAN = "local_feature_regularity_raw_v1"
OUT = ROOT / "outputs" / PLAN
QNN_CACHE = ROOT / "outputs" / "qnn_multiseed_all_paths_v1" / "features"
METHODS = ("zero", "selected_mrbi", "hybrid_ls", "hybrid_random", "multistart5")
KS = (1, 3, 5)
EPS = 1e-12


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_qnn_input_cache(dataset: str, seed: int):
    path = QNN_CACHE / f"seed{seed}" / f"{dataset}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the all-path QNN feature-cache job first."
        )
    with np.load(path, allow_pickle=False) as d:
        needed = {"Xtr_pca", "Xte_pca", "yte"}
        missing = needed.difference(d.files)
        if missing:
            raise RuntimeError(f"Cache schema missing {sorted(missing)}: {path}")
        return (
            np.asarray(d["Xtr_pca"], dtype=np.float64),
            np.asarray(d["Xte_pca"], dtype=np.float64),
            np.asarray(d["yte"], dtype=np.int64),
        )


def make_layer(cfg, Xtr_pca):
    layer_seed = cfg.seed + 1000 + cfg.latent_dim + int(100 * cfg.spectral_radius)
    return bench.make_random_implicit_layer(
        dx=Xtr_pca.shape[1], d=cfg.latent_dim, cfg=cfg, seed=layer_seed
    )


def selected_test_raw(Xte, layer, cfg_sel, row):
    if row["kind"] == "forced":
        Z, stats = bench.solve_forced_mrbi_features(
            Xte, layer, cfg_sel, cfg_sel.seed + 3000
        )
    elif row["kind"] == "hybrid":
        Z, stats = bench.solve_hybrid_features(
            Xte, layer, cfg_sel, cfg_sel.seed + 5000
        )
    else:
        raise RuntimeError(f"Unknown selected kind: {row['kind']}")
    return np.asarray(Z, dtype=np.float64), stats


def reconstruct_raw_test(dataset, seed, dataset_index, lock, cache_path):
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as d:
            arrays = {k: np.asarray(d[k]) for k in d.files}
        required = {"Xte_pca", "yte"} | {f"Zte_{m}" for m in METHODS}
        if set(arrays) != required:
            raise RuntimeError(f"Unexpected raw cache schema: {cache_path}")
        return arrays

    cfg = confirm.make_cfg(seed, lock)
    row, cfg_sel = confirm.selected_cfg(dataset, cfg, lock)
    Xtr_pca, Xte_pca, yte = load_qnn_input_cache(dataset, seed)
    layer = make_layer(cfg, Xtr_pca)

    print(f"RAW_START seed={seed} dataset={dataset}", flush=True)

    Zte_zero, _ = bench.solve_zero_features(Xte_pca, layer, cfg)
    Zte_sel, _ = selected_test_raw(Xte_pca, layer, cfg_sel, row)
    Zte_ls, _ = ls_control.solve_ls_policy(Xte_pca, layer, cfg_sel, row)
    Zte_random, _ = random_control.solve_random_policy(
        Xte_pca, layer, cfg_sel, row, seed + 9000 + 100 * dataset_index
    )
    ms_cfg = lock["confirmation_plan"]["random_multistart"]
    Zte_ms, _ = confirm.solve_multistart(
        Xte_pca,
        layer,
        cfg,
        rng_seed=seed + 7000 + 100 * dataset_index,
        n_starts=int(ms_cfg["n_random_starts"]),
        radius=1.0,
    )

    arrays = {
        "Xte_pca": np.asarray(Xte_pca, dtype=np.float64),
        "yte": np.asarray(yte, dtype=np.int64),
        "Zte_zero": np.asarray(Zte_zero, dtype=np.float64),
        "Zte_selected_mrbi": np.asarray(Zte_sel, dtype=np.float64),
        "Zte_hybrid_ls": np.asarray(Zte_ls, dtype=np.float64),
        "Zte_hybrid_random": np.asarray(Zte_random, dtype=np.float64),
        "Zte_multistart5": np.asarray(Zte_ms, dtype=np.float64),
    }
    if any(not np.all(np.isfinite(v)) for k, v in arrays.items() if k.startswith("Zte_")):
        raise RuntimeError(f"Non-finite raw features: {dataset} seed={seed}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(cache_path)
    print(f"RAW_DONE seed={seed} dataset={dataset}", flush=True)
    return arrays


def drel(a, b):
    num = np.linalg.norm(a - b, axis=-1)
    den = 1.0 + np.linalg.norm(a, axis=-1) + np.linalg.norm(b, axis=-1)
    return num / den


def knn_pairs(X, k):
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n <= k:
        raise RuntimeError(f"Need n>k, got n={n}, k={k}")
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    neigh = np.argsort(D, axis=1)[:, :k]
    pairs = sorted({tuple(sorted((i, int(j)))) for i in range(n) for j in neigh[i]})
    return np.asarray(pairs, dtype=np.int64)


def metrics_for_pairs(X, Z, pairs):
    i, j = pairs[:, 0], pairs[:, 1]
    dx = np.linalg.norm(X[i] - X[j], axis=1)
    dz = np.linalg.norm(Z[i] - Z[j], axis=1)
    dr = drel(Z[i], Z[j])
    amp = dz / (dx + EPS)
    ramp = dr / (dx + EPS)
    return {
        "n_pairs": int(len(pairs)),
        "median_input_distance": float(np.median(dx)),
        "median_feature_distance": float(np.median(dz)),
        "p90_feature_distance": float(np.quantile(dz, 0.90)),
        "p95_feature_distance": float(np.quantile(dz, 0.95)),
        "median_relative_feature_distance": float(np.median(dr)),
        "p90_relative_feature_distance": float(np.quantile(dr, 0.90)),
        "p95_relative_feature_distance": float(np.quantile(dr, 0.95)),
        "median_amplification": float(np.median(amp)),
        "p90_amplification": float(np.quantile(amp, 0.90)),
        "p95_amplification": float(np.quantile(amp, 0.95)),
        "median_relative_amplification": float(np.median(ramp)),
        "p90_relative_amplification": float(np.quantile(ramp, 0.90)),
        "p95_relative_amplification": float(np.quantile(ramp, 0.95)),
        "_dz": dz,
        "_dr": dr,
    }


def mean_rows(rows, group_keys, value_keys):
    groups = {}
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        groups.setdefault(key, []).append(r)
    out = []
    for key, vals in groups.items():
        row = {k: v for k, v in zip(group_keys, key)}
        for col in value_keys:
            a = np.asarray([float(v[col]) for v in vals], dtype=float)
            row[col] = float(np.mean(a))
        row["n_cases"] = len(vals)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true", help="run the diagnostic")
    ap.add_argument("--k", type=int, nargs="*", default=list(KS))
    args = ap.parse_args()
    if not args.run:
        raise SystemExit("Use --run")

    lock = confirm.load_lock()
    datasets = list(lock["development_evidence"]["datasets"])
    seeds = list(lock["confirmation_plan"]["seeds"])
    ks = tuple(sorted(set(int(k) for k in args.k)))

    case_rows = []
    pair_rows = []
    metric_cols = [
        "median_feature_distance", "p90_feature_distance", "p95_feature_distance",
        "median_relative_feature_distance", "p90_relative_feature_distance", "p95_relative_feature_distance",
        "median_amplification", "p90_amplification", "p95_amplification",
        "median_relative_amplification", "p90_relative_amplification", "p95_relative_amplification",
    ]

    for seed in seeds:
        for dataset_index, dataset in enumerate(datasets):
            cache = OUT / "raw_features" / f"seed{seed}" / f"{dataset}.npz"
            a = reconstruct_raw_test(dataset, seed, dataset_index, lock, cache)
            X = a["Xte_pca"]
            reps = {m: a[f"Zte_{m}"] for m in METHODS}

            for k in ks:
                pairs = knn_pairs(X, k)
                per_method = {}
                for method in METHODS:
                    mm = metrics_for_pairs(X, reps[method], pairs)
                    per_method[method] = mm
                    row = {
                        "dataset": dataset,
                        "seed": seed,
                        "k": k,
                        "method": method,
                        "n_test": len(X),
                        "n_pairs": mm["n_pairs"],
                        "median_input_distance": mm["median_input_distance"],
                    }
                    row.update({c: mm[c] for c in metric_cols})
                    case_rows.append(row)

                dz0 = per_method["zero"]["_dz"]
                dr0 = per_method["zero"]["_dr"]
                for method in METHODS[1:]:
                    dz = per_method[method]["_dz"]
                    dr = per_method[method]["_dr"]
                    log_dz = np.log((dz + EPS) / (dz0 + EPS))
                    log_dr = np.log((dr + EPS) / (dr0 + EPS))
                    pair_rows.append({
                        "dataset": dataset,
                        "seed": seed,
                        "k": k,
                        "method": method,
                        "median_log_feature_distance_ratio_vs_zero": float(np.median(log_dz)),
                        "mean_log_feature_distance_ratio_vs_zero": float(np.mean(log_dz)),
                        "median_log_relative_distance_ratio_vs_zero": float(np.median(log_dr)),
                        "mean_log_relative_distance_ratio_vs_zero": float(np.mean(log_dr)),
                        "fraction_neighbor_pairs_farther_than_zero": float(np.mean(dz > dz0 + 1e-12)),
                    })

            print(f"REGULARITY_DONE seed={seed} dataset={dataset}", flush=True)

    write_csv(OUT / "case_summary.csv", case_rows)
    write_csv(OUT / "pairwise_vs_zero.csv", pair_rows)

    dataset_rows = mean_rows(
        case_rows,
        group_keys=("dataset", "k", "method"),
        value_keys=tuple(metric_cols),
    )
    write_csv(OUT / "dataset_summary.csv", dataset_rows)

    overall_rows = mean_rows(
        dataset_rows,
        group_keys=("k", "method"),
        value_keys=tuple(metric_cols),
    )
    write_csv(OUT / "overall_summary.csv", overall_rows)

    pair_dataset_rows = mean_rows(
        pair_rows,
        group_keys=("dataset", "k", "method"),
        value_keys=(
            "median_log_feature_distance_ratio_vs_zero",
            "mean_log_feature_distance_ratio_vs_zero",
            "median_log_relative_distance_ratio_vs_zero",
            "mean_log_relative_distance_ratio_vs_zero",
            "fraction_neighbor_pairs_farther_than_zero",
        ),
    )
    pair_overall_rows = mean_rows(
        pair_dataset_rows,
        group_keys=("k", "method"),
        value_keys=(
            "median_log_feature_distance_ratio_vs_zero",
            "mean_log_feature_distance_ratio_vs_zero",
            "median_log_relative_distance_ratio_vs_zero",
            "mean_log_relative_distance_ratio_vs_zero",
            "fraction_neighbor_pairs_farther_than_zero",
        ),
    )

    summary = {
        "plan": PLAN,
        "analysis_role": "post-confirmation descriptive mechanism diagnostic",
        "rho": 2.0,
        "input_space": "bounded four-dimensional PCA test representation",
        "feature_space": "raw unstandardized returned equilibrium states",
        "neighbor_graph": "shared test-set kNN graph reused across selectors",
        "k_values": list(ks),
        "methods": list(METHODS),
        "datasets": datasets,
        "seeds": seeds,
        "overall": overall_rows,
        "pairwise_vs_zero_overall": pair_overall_rows,
        "interpretation_guardrail": (
            "Lower local amplification is consistent with a locally more regular returned-state map. "
            "This diagnostic does not reconstruct global solution branches and is not a causal test of readout generalization."
        ),
    }
    atomic_text(OUT / "summary.json", json.dumps(summary, indent=2) + "\n")

    print(f"LOCAL_FEATURE_REGULARITY_OK {OUT}", flush=True)
    for r in overall_rows:
        if int(r["k"]) == 3:
            print(
                "K3 "
                f"method={r['method']} "
                f"med_rel_amp={r['median_relative_amplification']:.6g} "
                f"p90_rel_amp={r['p90_relative_amplification']:.6g} "
                f"p95_amp={r['p95_amplification']:.6g} "
                f"p95_rel_amp={r['p95_relative_amplification']:.6g}",
                flush=True,
            )


if __name__ == "__main__":
    main()
