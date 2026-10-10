"""Repeat the QNN readout over several seeds for all implicit feature paths.

The feature-generating numerical path is held fixed within each dataset/data-seed
case. Only the QNN initialization and minibatch order change. The script caches
standardized features so expensive implicit solves are performed at most once.

Methods:
    zero
    selected_mrbi
    hybrid_ls
    hybrid_random
    multistart5

If the rho=2.0 anchor cache exists, its Zero and selected-MRBI features are reused.
Otherwise those two paths are reconstructed with the locked confirmation protocol.
Hybrid-LS, Hybrid-Random, and multistart-5 are reconstructed once and cached.
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

import main_qnn_benchmark as bench
import run_locked_selected_profile_confirmation as confirm
import run_hybrid_random_control as random_control
import run_least_squares_control as ls_control

PLAN = "qnn_multiseed_all_paths_v1"
DEFAULT_OUT = ROOT / "outputs" / PLAN
RHO_CACHE = ROOT / "outputs" / "fixed_profile_rho_sensitivity_v1" / "rho_2p00"
CONFIRM_CSV = ROOT / "results" / "final" / "confirmation_dataset_summary.csv"
RANDOM_CSV = ROOT / "results" / "final" / "hybrid_random_control_dataset_summary.csv"
LS_CSV = ROOT / "results" / "final" / "hybrid_least_squares_control_dataset_summary.csv"

METHODS = ("zero", "selected_mrbi", "hybrid_ls", "hybrid_random", "multistart5")
PAIRS = (
    ("selected_mrbi", "zero"),
    ("selected_mrbi", "hybrid_ls"),
    ("selected_mrbi", "hybrid_random"),
    ("selected_mrbi", "multistart5"),
    ("multistart5", "zero"),
)
TOL = 1e-10


def read_csv_by_dataset(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return {row["dataset"]: row for row in csv.DictReader(f)}


def exact_signed_rank(values, tol=1e-12):
    values = np.asarray(values, dtype=float)
    values = values[np.abs(values) > tol]
    if len(values) == 0:
        return {
            "n_nonzero": 0,
            "statistic_wplus": 0.0,
            "p_greater": 1.0,
            "p_two_sided": 1.0,
        }
    ranks = rankdata(np.abs(values), method="average")
    observed = float(np.sum(ranks[values > 0]))
    null = np.fromiter(
        (
            sum(rank for rank, bit in zip(ranks, bits) if bit)
            for bits in product((0, 1), repeat=len(ranks))
        ),
        dtype=float,
    )
    p_greater = float(np.mean(null >= observed - 1e-15))
    p_less = float(np.mean(null <= observed + 1e-15))
    return {
        "n_nonzero": int(len(values)),
        "statistic_wplus": observed,
        "p_greater": p_greater,
        "p_two_sided": float(min(1.0, 2.0 * min(p_greater, p_less))),
    }


def prepare_from_scratch(dataset, data_seed, dataset_index, lock):
    cfg = confirm.make_cfg(data_seed, lock)
    row, cfg_sel = confirm.selected_cfg(dataset, cfg, lock)

    X, y = bench.load_dataset(dataset, cfg)
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        stratify=y,
        random_state=cfg.seed,
    )
    Xtr_p, Xte_p = bench.preprocess_to_pca(
        Xtr_raw, Xte_raw, cfg.pca_dim, cfg.seed
    )
    layer_seed = cfg.seed + 1000 + cfg.latent_dim + int(100 * cfg.spectral_radius)
    layer = bench.make_random_implicit_layer(
        dx=Xtr_p.shape[1],
        d=cfg.latent_dim,
        cfg=cfg,
        seed=layer_seed,
    )
    return cfg, row, cfg_sel, Xtr_p, Xte_p, ytr, yte, layer


def layer_from_pca(cfg, Xtr_p):
    layer_seed = cfg.seed + 1000 + cfg.latent_dim + int(100 * cfg.spectral_radius)
    return bench.make_random_implicit_layer(
        dx=Xtr_p.shape[1],
        d=cfg.latent_dim,
        cfg=cfg,
        seed=layer_seed,
    )


def load_anchor(dataset, data_seed):
    path = RHO_CACHE / f"seed{data_seed}" / "features" / f"{dataset}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: np.asarray(data[k]) for k in data.files}
    required = {
        "Xtr_pca", "Xte_pca", "Ztr_zero", "Zte_zero",
        "Ztr_selected", "Zte_selected", "ytr", "yte",
    }
    if set(arrays) != required:
        raise RuntimeError(f"Unexpected rho=2.0 cache schema: {path}")
    return arrays


def build_feature_cache(dataset, data_seed, dataset_index, lock, cache_path):
    cfg = confirm.make_cfg(data_seed, lock)
    row, cfg_sel = confirm.selected_cfg(dataset, cfg, lock)

    anchor = load_anchor(dataset, data_seed)
    if anchor is not None:
        Xtr_p = anchor["Xtr_pca"].astype(np.float64)
        Xte_p = anchor["Xte_pca"].astype(np.float64)
        ytr = anchor["ytr"].astype(int)
        yte = anchor["yte"].astype(int)
        Ztr_zero = anchor["Ztr_zero"].astype(np.float64)
        Zte_zero = anchor["Zte_zero"].astype(np.float64)
        Ztr_sel = anchor["Ztr_selected"].astype(np.float64)
        Zte_sel = anchor["Zte_selected"].astype(np.float64)
        layer = layer_from_pca(cfg, Xtr_p)
        zero_selected_source = "rho_2p00_anchor_cache"
    else:
        (
            cfg, row, cfg_sel, Xtr_p, Xte_p, ytr, yte, layer
        ) = prepare_from_scratch(dataset, data_seed, dataset_index, lock)

        print(
            f"FEATURE_REBUILD_SLOW dataset={dataset} seed={data_seed} "
            "reason=missing_rho2_anchor_cache",
            flush=True,
        )
        Ztr_zero_raw, _ = bench.solve_zero_features(Xtr_p, layer, cfg)
        Zte_zero_raw, _ = bench.solve_zero_features(Xte_p, layer, cfg)
        Ztr_sel_raw, Zte_sel_raw, _, _ = confirm.solve_selected(
            Xtr_p, Xte_p, layer, cfg_sel, row
        )
        Ztr_zero, Zte_zero = bench.standardize_pair(Ztr_zero_raw, Zte_zero_raw)
        Ztr_sel, Zte_sel = bench.standardize_pair(Ztr_sel_raw, Zte_sel_raw)
        zero_selected_source = "reconstructed_locked_protocol"

    Ztr_random_raw, _ = random_control.solve_random_policy(
        Xtr_p,
        layer,
        cfg_sel,
        row,
        data_seed + 8000 + 100 * dataset_index,
    )
    Zte_random_raw, _ = random_control.solve_random_policy(
        Xte_p,
        layer,
        cfg_sel,
        row,
        data_seed + 9000 + 100 * dataset_index,
    )
    Ztr_random, Zte_random = bench.standardize_pair(
        Ztr_random_raw, Zte_random_raw
    )

    Ztr_ls_raw, _ = ls_control.solve_ls_policy(Xtr_p, layer, cfg_sel, row)
    Zte_ls_raw, _ = ls_control.solve_ls_policy(Xte_p, layer, cfg_sel, row)
    Ztr_ls, Zte_ls = bench.standardize_pair(Ztr_ls_raw, Zte_ls_raw)

    ms_cfg = lock["confirmation_plan"]["random_multistart"]
    Ztr_ms_raw, _ = confirm.solve_multistart(
        Xtr_p,
        layer,
        cfg,
        rng_seed=data_seed + 6000 + 100 * dataset_index,
        n_starts=int(ms_cfg["n_random_starts"]),
        radius=1.0,
    )
    Zte_ms_raw, _ = confirm.solve_multistart(
        Xte_p,
        layer,
        cfg,
        rng_seed=data_seed + 7000 + 100 * dataset_index,
        n_starts=int(ms_cfg["n_random_starts"]),
        radius=1.0,
    )
    Ztr_ms, Zte_ms = bench.standardize_pair(Ztr_ms_raw, Zte_ms_raw)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(
            f,
            ytr=np.asarray(ytr, dtype=np.int64),
            yte=np.asarray(yte, dtype=np.int64),
            Xtr_pca=np.asarray(Xtr_p, dtype=np.float64),
            Xte_pca=np.asarray(Xte_p, dtype=np.float64),
            Ztr_zero=np.asarray(Ztr_zero, dtype=np.float64),
            Zte_zero=np.asarray(Zte_zero, dtype=np.float64),
            Ztr_selected_mrbi=np.asarray(Ztr_sel, dtype=np.float64),
            Zte_selected_mrbi=np.asarray(Zte_sel, dtype=np.float64),
            Ztr_hybrid_ls=np.asarray(Ztr_ls, dtype=np.float64),
            Zte_hybrid_ls=np.asarray(Zte_ls, dtype=np.float64),
            Ztr_hybrid_random=np.asarray(Ztr_random, dtype=np.float64),
            Zte_hybrid_random=np.asarray(Zte_random, dtype=np.float64),
            Ztr_multistart5=np.asarray(Ztr_ms, dtype=np.float64),
            Zte_multistart5=np.asarray(Zte_ms, dtype=np.float64),
        )
    tmp.replace(cache_path)

    meta = {
        "dataset": dataset,
        "data_seed": int(data_seed),
        "selected_method": row["method"],
        "zero_selected_source": zero_selected_source,
    }
    cache_path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return load_feature_cache(cache_path)


def load_feature_cache(path):
    with np.load(path, allow_pickle=False) as data:
        a = {k: np.asarray(data[k]) for k in data.files}
    reps = {
        "zero": (a["Ztr_zero"], a["Zte_zero"]),
        "selected_mrbi": (
            a["Ztr_selected_mrbi"], a["Zte_selected_mrbi"]
        ),
        "hybrid_ls": (a["Ztr_hybrid_ls"], a["Zte_hybrid_ls"]),
        "hybrid_random": (
            a["Ztr_hybrid_random"], a["Zte_hybrid_random"]
        ),
        "multistart5": (
            a["Ztr_multistart5"], a["Zte_multistart5"]
        ),
    }
    return a["ytr"].astype(int), a["yte"].astype(int), reps


def canonical_expected():
    conf = read_csv_by_dataset(CONFIRM_CSV)
    rnd = read_csv_by_dataset(RANDOM_CSV)
    ls = read_csv_by_dataset(LS_CSV)
    out = {}
    for dataset in conf:
        out[dataset] = {
            "zero": float(conf[dataset]["zero"]),
            "selected_mrbi": float(conf[dataset]["selected_mrbi"]),
            "multistart5": float(conf[dataset]["multistart5"]),
            "hybrid_random": float(rnd[dataset]["hybrid_random_ba"]),
            "hybrid_ls": float(ls[dataset]["hybrid_ls_ba"]),
        }
    return out


def summarize(rows, datasets, replicates):
    by_dataset_method = []
    for dataset in datasets:
        for method in METHODS:
            rep_means = []
            for rep in range(replicates):
                vals = [
                    r["balanced_accuracy"]
                    for r in rows
                    if r["dataset"] == dataset
                    and r["method"] == method
                    and r["qnn_rep"] == rep
                ]
                if len(vals) != 10:
                    raise RuntimeError(
                        f"Expected 10 data seeds for {dataset}/{method}/rep{rep}, "
                        f"found {len(vals)}"
                    )
                rep_means.append(float(np.mean(vals)))
            arr = np.asarray(rep_means, dtype=float)
            by_dataset_method.append({
                "dataset": dataset,
                "method": method,
                "replicate_mean_ba": rep_means,
                "mean_over_qnn_seeds": float(np.mean(arr)),
                "sd_across_qnn_seed_means": float(np.std(arr, ddof=1)),
                "range_across_qnn_seed_means": float(np.ptp(arr)),
            })

    expected = canonical_expected()
    canonical_checks = []
    for row in by_dataset_method:
        observed = float(row["replicate_mean_ba"][0])
        target = float(expected[row["dataset"]][row["method"]])
        ok = bool(np.isclose(observed, target, atol=TOL, rtol=0.0))
        canonical_checks.append({
            "dataset": row["dataset"],
            "method": row["method"],
            "observed": observed,
            "expected": target,
            "difference": observed - target,
            "passed": ok,
        })
    failed = [r for r in canonical_checks if not r["passed"]]
    if failed:
        raise RuntimeError(
            "Canonical QNN reproduction failed: "
            + json.dumps(failed[:5], indent=2)
        )

    pair_summary = {}
    for a, b in PAIRS:
        per_rep = []
        dataset_ensemble_deltas = []
        for rep in range(replicates):
            ds_deltas = []
            for dataset in datasets:
                ra = next(
                    r for r in by_dataset_method
                    if r["dataset"] == dataset and r["method"] == a
                )
                rb = next(
                    r for r in by_dataset_method
                    if r["dataset"] == dataset and r["method"] == b
                )
                ds_deltas.append(
                    float(ra["replicate_mean_ba"][rep] - rb["replicate_mean_ba"][rep])
                )
            per_rep.append({
                "qnn_rep": rep,
                "dataset_deltas": ds_deltas,
                "mean_delta": float(np.mean(ds_deltas)),
                "median_delta": float(np.median(ds_deltas)),
                "positive": int(np.sum(np.asarray(ds_deltas) > 1e-12)),
                "neutral": int(np.sum(np.abs(np.asarray(ds_deltas)) <= 1e-12)),
                "negative": int(np.sum(np.asarray(ds_deltas) < -1e-12)),
                "signed_rank": exact_signed_rank(ds_deltas),
            })

        for dataset in datasets:
            ra = next(
                r for r in by_dataset_method
                if r["dataset"] == dataset and r["method"] == a
            )
            rb = next(
                r for r in by_dataset_method
                if r["dataset"] == dataset and r["method"] == b
            )
            dataset_ensemble_deltas.append(
                float(ra["mean_over_qnn_seeds"] - rb["mean_over_qnn_seeds"])
            )

        arr = np.asarray(dataset_ensemble_deltas, dtype=float)
        pair_summary[f"{a}_minus_{b}"] = {
            "per_qnn_rep": per_rep,
            "dataset_ensemble_deltas": dataset_ensemble_deltas,
            "ensemble_mean_delta": float(np.mean(arr)),
            "ensemble_median_delta": float(np.median(arr)),
            "positive": int(np.sum(arr > 1e-12)),
            "neutral": int(np.sum(np.abs(arr) <= 1e-12)),
            "negative": int(np.sum(arr < -1e-12)),
            "signed_rank": exact_signed_rank(arr),
        }

    return {
        "replicates": int(replicates),
        "methods": list(METHODS),
        "n_dataset_seed_cases": 90,
        "n_qnn_trainings": int(90 * replicates * len(METHODS)),
        "canonical_reproduction": {
            "tolerance": TOL,
            "passed": True,
            "checks": canonical_checks,
        },
        "dataset_method_summary": by_dataset_method,
        "pair_summary": pair_summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replicates",
        type=int,
        default=4,
        help="QNN seeds per fixed feature set, including the canonical seed",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.replicates < 2:
        raise SystemExit("Use at least two QNN seeds")
    if not bench.HAS_QNN:
        raise SystemExit(f"QNN dependency unavailable: {bench.QNN_IMPORT_ERROR}")

    lock = confirm.load_lock()
    datasets = list(lock["development_evidence"]["datasets"])
    data_seeds = list(lock["confirmation_plan"]["seeds"])

    out_dir = args.out_dir.resolve()
    feature_dir = out_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "checkpoint.json"

    rows = []
    if checkpoint.exists():
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        if int(payload.get("replicates", -1)) != args.replicates:
            raise SystemExit(
                "Existing checkpoint uses a different --replicates value. "
                "Use a new --out-dir or remove the old checkpoint."
            )
        rows = payload.get("rows", [])

    completed = {
        (
            r["dataset"],
            int(r["data_seed"]),
            r["method"],
            int(r["qnn_rep"]),
        )
        for r in rows
    }

    for data_seed in data_seeds:
        for dataset_index, dataset in enumerate(datasets):
            cache_path = feature_dir / f"seed{data_seed}" / f"{dataset}.npz"
            if cache_path.exists():
                ytr, yte, reps = load_feature_cache(cache_path)
            else:
                print(
                    f"FEATURE_START dataset={dataset} seed={data_seed}",
                    flush=True,
                )
                ytr, yte, reps = build_feature_cache(
                    dataset,
                    data_seed,
                    dataset_index,
                    lock,
                    cache_path,
                )
                print(
                    f"FEATURE_DONE dataset={dataset} seed={data_seed}",
                    flush=True,
                )

            cfg = confirm.make_cfg(data_seed, lock)
            for rep in range(args.replicates):
                qnn_seed = data_seed + 777 + rep * 10000
                for method in METHODS:
                    key = (dataset, data_seed, method, rep)
                    if key in completed:
                        continue
                    Xtr, Xte = reps[method]
                    print(
                        f"QNN_START dataset={dataset} data_seed={data_seed} "
                        f"rep={rep} method={method}",
                        flush=True,
                    )
                    metrics = bench.train_qnn(
                        Xtr, ytr, Xte, yte, cfg, qnn_seed
                    )
                    row = {
                        "dataset": dataset,
                        "data_seed": int(data_seed),
                        "method": method,
                        "qnn_rep": int(rep),
                        "qnn_seed": int(qnn_seed),
                        "balanced_accuracy": float(
                            metrics["balanced_accuracy"]
                        ),
                    }
                    rows.append(row)
                    completed.add(key)
                    checkpoint.write_text(
                        json.dumps(
                            {
                                "plan": PLAN,
                                "replicates": int(args.replicates),
                                "rows": rows,
                            },
                            indent=2,
                            allow_nan=False,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    print(
                        f"QNN_DONE dataset={dataset} data_seed={data_seed} "
                        f"rep={rep} method={method} "
                        f"ba={row['balanced_accuracy']:.4f}",
                        flush=True,
                    )

    summary = summarize(rows, datasets, args.replicates)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {"plan": PLAN, "summary": summary},
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("QNN_MULTISEED_ALL_PATHS_OK", flush=True)
    for key, value in summary["pair_summary"].items():
        print(
            f"{key} ensemble_mean_delta={value['ensemble_mean_delta']:+.6f} "
            f"p_greater={value['signed_rank']['p_greater']:.6f} "
            f"p_two_sided={value['signed_rank']['p_two_sided']:.6f}",
            flush=True,
        )
    print(f"SUMMARY {summary_path}", flush=True)


if __name__ == "__main__":
    main()
