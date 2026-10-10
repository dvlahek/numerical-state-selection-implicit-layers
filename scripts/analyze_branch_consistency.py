from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

import main_qnn_benchmark as bench
import mrbi
import run_locked_selected_profile_confirmation as confirm
import run_hybrid_random_control as random_control
import run_least_squares_control as ls_control

PLAN = "branch_consistency_analysis_v1"
DEFAULT_OUT = ROOT / "outputs" / PLAN
THRESHOLDS = (1e-6, 1e-4)
EPS = 1e-12


def rel_distance(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (1.0 + np.linalg.norm(a) + np.linalg.norm(b)))


def same_rate(A, B, tol):
    vals = np.asarray([rel_distance(a, b) for a, b in zip(A, B)], dtype=float)
    return float(np.mean(vals <= tol)), vals


def nearest_neighbor_smoothness(X, Z):
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    n = len(X)
    if n < 2:
        return {"mean": None, "median": None, "q25": None, "q75": None}
    ratios = []
    for i in range(n):
        dx = np.linalg.norm(X - X[i], axis=1)
        dx[i] = np.inf
        j = int(np.argmin(dx))
        ratios.append(float(np.linalg.norm(Z[i] - Z[j]) / (dx[j] + EPS)))
    arr = np.asarray(ratios, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
    }


def cluster_count(states, tol=1e-6):
    reps = []
    for state in states:
        if not any(rel_distance(state, rep) <= tol for rep in reps):
            reps.append(np.asarray(state, dtype=np.float64))
    return len(reps)


def solve_multistart_detailed(X, layer, cfg, rng_seed, n_starts=5, radius=1.0):
    F, J = mrbi.make_residual_and_jacobian(layer)
    root_cfg = bench.make_root_cfg(cfg)
    rng = np.random.default_rng(rng_seed)
    z0 = np.zeros(layer.d, dtype=np.float64)

    selected = []
    details = []
    for pos, x in enumerate(np.asarray(X, dtype=np.float64)):
        zero = mrbi.solve_root(F, J, x, z0, cfg=root_cfg)
        others = []
        for _ in range(n_starts):
            start = rng.uniform(-radius, radius, size=layer.d)
            others.append(mrbi.solve_root(F, J, x, start, cfg=root_cfg))
        best = confirm.choose_multistart(zero, others)
        selected.append(best.z_star)

        successful = [
            r for r in [zero] + others
            if r.success and np.isfinite(r.residual) and np.all(np.isfinite(r.z_star))
        ]
        details.append({
            "position": int(pos),
            "strict_success_count": int(len(successful)),
            "distinct_strict_roots_1e6": int(cluster_count([r.z_star for r in successful], 1e-6)) if successful else 0,
            "distinct_strict_roots_1e4": int(cluster_count([r.z_star for r in successful], 1e-4)) if successful else 0,
            "selected_same_as_zero_1e6": int(rel_distance(best.z_star, zero.z_star) <= 1e-6),
            "selected_same_as_zero_1e4": int(rel_distance(best.z_star, zero.z_star) <= 1e-4),
            "zero_success": int(zero.success),
            "selected_success": int(best.success),
        })
    return np.asarray(selected, dtype=np.float64), details


def solve_selected_test(Xte, layer, cfg_sel, row, seed):
    if row["kind"] == "forced":
        Z, stats = bench.solve_forced_mrbi_features(Xte, layer, cfg_sel, seed + 3000)
    else:
        Z, stats = bench.solve_hybrid_features(Xte, layer, cfg_sel, seed + 5000)
    return np.asarray(Z, dtype=np.float64), stats


def prepare(dataset, seed, lock):
    cfg = confirm.make_cfg(seed, lock)
    row, cfg_sel = confirm.selected_cfg(dataset, cfg, lock)
    X, y = bench.load_dataset(dataset, cfg)
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        stratify=y,
        random_state=cfg.seed,
    )
    Xtr_p, Xte_p = bench.preprocess_to_pca(Xtr_raw, Xte_raw, cfg.pca_dim, cfg.seed)
    layer_seed = cfg.seed + 1000 + cfg.latent_dim + int(100 * cfg.spectral_radius)
    layer = bench.make_random_implicit_layer(
        dx=Xtr_p.shape[1], d=cfg.latent_dim, cfg=cfg, seed=layer_seed
    )
    return cfg, row, cfg_sel, Xtr_p, Xte_p, ytr, yte, layer


def run_one(dataset, seed, dataset_index, lock):
    cfg, row, cfg_sel, Xtr, Xte, ytr, yte, layer = prepare(dataset, seed, lock)

    Z_zero, _ = bench.solve_zero_features(Xte, layer, cfg)

    t0 = time.perf_counter()
    Z_mrbi, _ = solve_selected_test(Xte, layer, cfg_sel, row, seed)
    mrbi_test_wall = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    _Ztr_random, random_train_stats = random_control.solve_random_policy(
        Xtr, layer, cfg_sel, row, seed + 8000 + 100 * dataset_index
    )
    Z_random, random_test_stats = random_control.solve_random_policy(
        Xte, layer, cfg_sel, row, seed + 9000 + 100 * dataset_index
    )
    random_feature_wall = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    _Ztr_ls, ls_train_stats = ls_control.solve_ls_policy(Xtr, layer, cfg_sel, row)
    Z_ls, ls_test_stats = ls_control.solve_ls_policy(Xte, layer, cfg_sel, row)
    ls_feature_wall = float(time.perf_counter() - t0)

    Z_ms, ms_details = solve_multistart_detailed(
        Xte,
        layer,
        cfg,
        seed + 7000 + 100 * dataset_index,
        n_starts=int(lock["confirmation_plan"]["random_multistart"]["n_random_starts"]),
        radius=1.0,
    )

    methods = {
        "zero": np.asarray(Z_zero),
        "mrbi": np.asarray(Z_mrbi),
        "hybrid_ls": np.asarray(Z_ls),
        "hybrid_random": np.asarray(Z_random),
        "multistart5": np.asarray(Z_ms),
    }

    pair_names = [
        ("mrbi", "zero"),
        ("hybrid_ls", "zero"),
        ("hybrid_random", "zero"),
        ("multistart5", "zero"),
        ("mrbi", "hybrid_ls"),
        ("mrbi", "hybrid_random"),
        ("hybrid_ls", "hybrid_random"),
    ]

    pair_summary = {}
    pair_arrays = {}
    for a, b in pair_names:
        rates = {}
        arr = None
        for tol in THRESHOLDS:
            rate, vals = same_rate(methods[a], methods[b], tol)
            rates[f"same_rate_{tol:g}"] = rate
            arr = vals
        pair_summary[f"{a}_vs_{b}"] = {
            **rates,
            "mean_rel_distance": float(np.mean(arr)),
            "median_rel_distance": float(np.median(arr)),
        }
        pair_arrays[f"{a}_vs_{b}"] = arr

    smoothness = {name: nearest_neighbor_smoothness(Xte, Z) for name, Z in methods.items()}

    distinct_1e6 = np.asarray([d["distinct_strict_roots_1e6"] for d in ms_details], dtype=float)
    distinct_1e4 = np.asarray([d["distinct_strict_roots_1e4"] for d in ms_details], dtype=float)

    sample_rows = []
    for i in range(len(Xte)):
        row_out = {
            "dataset": dataset,
            "seed": int(seed),
            "position": int(i),
            "label": int(yte[i]),
            "multistart_distinct_strict_roots_1e6": int(distinct_1e6[i]),
            "multistart_distinct_strict_roots_1e4": int(distinct_1e4[i]),
            "multistart_selected_same_zero_1e6": int(ms_details[i]["selected_same_as_zero_1e6"]),
            "multistart_selected_same_zero_1e4": int(ms_details[i]["selected_same_as_zero_1e4"]),
        }
        for pair, vals in pair_arrays.items():
            row_out[f"rel_distance_{pair}"] = float(vals[i])
        sample_rows.append(row_out)

    return {
        "dataset": dataset,
        "seed": int(seed),
        "selected_method": row["method"],
        "selected_kind": row["kind"],
        "n_test": int(len(Xte)),
        "pair_summary": pair_summary,
        "smoothness": smoothness,
        "multistart": {
            "mean_distinct_strict_roots_1e6": float(np.mean(distinct_1e6)),
            "mean_distinct_strict_roots_1e4": float(np.mean(distinct_1e4)),
            "fraction_multiple_strict_roots_1e6": float(np.mean(distinct_1e6 >= 2)),
            "fraction_multiple_strict_roots_1e4": float(np.mean(distinct_1e4 >= 2)),
            "selected_same_zero_1e6": float(np.mean([d["selected_same_as_zero_1e6"] for d in ms_details])),
            "selected_same_zero_1e4": float(np.mean([d["selected_same_as_zero_1e4"] for d in ms_details])),
        },
        "runtime": {
            "mrbi_test_wall_sec": mrbi_test_wall,
            "hybrid_random_feature_wall_sec": random_feature_wall,
            "hybrid_ls_feature_wall_sec": ls_feature_wall,
            "hybrid_random_train_mean_sec_per_sample": float(random_train_stats["mean_runtime_sec_per_sample"]),
            "hybrid_random_test_mean_sec_per_sample": float(random_test_stats["mean_runtime_sec_per_sample"]),
            "hybrid_ls_train_mean_sec_per_sample": float(ls_train_stats["mean_runtime_sec_per_sample"]),
            "hybrid_ls_test_mean_sec_per_sample": float(ls_test_stats["mean_runtime_sec_per_sample"]),
        },
        "sample_rows": sample_rows,
    }


def aggregate(rows):
    pair_keys = list(rows[0]["pair_summary"].keys())
    out = {"n_dataset_seed_runs": len(rows), "pairs": {}, "smoothness": {}, "multistart": {}, "runtime": {}}
    for pair in pair_keys:
        out["pairs"][pair] = {}
        for key in rows[0]["pair_summary"][pair]:
            vals = [r["pair_summary"][pair][key] for r in rows]
            out["pairs"][pair][key] = float(np.mean(vals))

    for method in rows[0]["smoothness"]:
        out["smoothness"][method] = {}
        for stat in ("mean", "median"):
            vals = [r["smoothness"][method][stat] for r in rows if r["smoothness"][method][stat] is not None]
            out["smoothness"][method][stat] = float(np.mean(vals))

    for key in rows[0]["multistart"]:
        out["multistart"][key] = float(np.mean([r["multistart"][key] for r in rows]))

    for key in ("hybrid_random_feature_wall_sec", "hybrid_ls_feature_wall_sec", "mrbi_test_wall_sec"):
        vals = [r["runtime"][key] for r in rows]
        out["runtime"][f"mean_{key}"] = float(np.mean(vals))
        out["runtime"][f"median_{key}"] = float(np.median(vals))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="Run one confirmation seed only")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    lock = confirm.load_lock()
    seeds = [args.seed] if args.seed is not None else list(lock["confirmation_plan"]["seeds"])
    for seed in seeds:
        if seed not in lock["confirmation_plan"]["seeds"]:
            raise SystemExit("Seed must be one of confirmation seeds 40-49")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.json"
    rows = []
    if checkpoint_path.exists():
        rows = json.loads(checkpoint_path.read_text(encoding="utf-8")).get("dataset_seed_rows", [])
    completed = {(r["dataset"], int(r["seed"])) for r in rows}

    datasets = list(lock["development_evidence"]["datasets"])
    for seed in seeds:
        for dataset_index, dataset in enumerate(datasets):
            if (dataset, seed) in completed:
                print(f"SKIP dataset={dataset} seed={seed}", flush=True)
                continue
            print(f"BRANCH_START dataset={dataset} seed={seed}", flush=True)
            result = run_one(dataset, seed, dataset_index, lock)
            rows.append(result)
            checkpoint_path.write_text(
                json.dumps({"plan": PLAN, "dataset_seed_rows": rows}, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            p = result["pair_summary"]
            print(
                "BRANCH_DONE "
                f"dataset={dataset} seed={seed} "
                f"same_mrbi_ls={p['mrbi_vs_hybrid_ls']['same_rate_1e-06']:.3f} "
                f"same_mrbi_zero={p['mrbi_vs_zero']['same_rate_1e-06']:.3f} "
                f"same_random_zero={p['hybrid_random_vs_zero']['same_rate_1e-06']:.3f}",
                flush=True,
            )

    selected_rows = [r for r in rows if int(r["seed"]) in seeds]
    overall = aggregate(selected_rows)

    sample_rows = []
    for r in selected_rows:
        sample_rows.extend(r.pop("sample_rows", []))

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps({"plan": PLAN, "thresholds": THRESHOLDS, "overall": overall, "dataset_seed_rows": selected_rows}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    sample_csv = out_dir / "sample_distances.csv"
    if sample_rows:
        with sample_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sample_rows)

    print("BRANCH_CONSISTENCY_ANALYSIS_OK")
    print(json.dumps(overall, indent=2))
    print(f"SUMMARY {summary_path}")
    print(f"SAMPLES {sample_csv}")


if __name__ == "__main__":
    main()
