from __future__ import annotations

import csv
import json
import time
from itertools import product
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_locked_selected_profile_confirmation as base

bench = base.bench
mrbi = base.mrbi

PLAN = "hybrid_least_squares_control_v1"
OUT = ROOT / "outputs" / PLAN
BASELINE = ROOT / "results" / "final" / "confirmation_dataset_summary.csv"
LS_MAX_NFEV = 300


def exact_signed_rank_greater(values, tol=1e-12):
    values = np.asarray(values, dtype=float)
    values = values[np.abs(values) > tol]
    if len(values) == 0:
        return 0.0, 1.0
    ranks = rankdata(np.abs(values), method="average")
    statistic = float(np.sum(ranks[values > 0]))
    null_stats = np.fromiter(
        (
            sum(rank for rank, bit in zip(ranks, bits) if bit)
            for bits in product((0, 1), repeat=len(ranks))
        ),
        dtype=float,
    )
    return statistic, float(np.mean(null_stats >= statistic - 1e-15))


def bootstrap_ci(values, seed=20261004, n_boot=20000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def build_ls_candidate(F, J, x, dim, radius):
    z0 = np.zeros(dim, dtype=np.float64)

    def fun(z):
        return np.asarray(F(z, x), dtype=np.float64)

    def jac(z):
        return np.asarray(J(z, x), dtype=np.float64)

    try:
        res = least_squares(
            fun,
            z0,
            jac=jac,
            bounds=(-radius, radius),
            method="trf",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=LS_MAX_NFEV,
        )
        z = np.asarray(res.x, dtype=np.float64)
        residual = float(np.linalg.norm(fun(z)))
        return z, {
            "success": bool(res.success),
            "residual": residual,
            "nfev": int(res.nfev),
            "status": int(res.status),
        }
    except Exception as exc:
        return z0, {
            "success": False,
            "residual": float("inf"),
            "nfev": 0,
            "status": -999,
            "error": repr(exc),
        }


def solve_ls_policy(X, layer, cfg, row):
    F, J = mrbi.make_residual_and_jacobian(layer)
    root_cfg = bench.make_root_cfg(cfg)
    hybrid_cfg = bench.make_hybrid_cfg(cfg) if row["kind"] == "hybrid" else None
    z_zero = np.zeros(layer.d, dtype=np.float64)
    radius = float(cfg.search_radius)

    Z = []
    success = []
    trigger = []
    accepted = []
    changed = []
    distances = []
    residuals = []
    candidate_residuals = []
    candidate_nfev = []
    runtimes = []

    for x in np.asarray(X, dtype=np.float64):
        t0 = time.perf_counter()
        zero = mrbi.solve_root(F, J, x, z_zero, cfg=root_cfg)
        s_zero, jac_zero_ok = mrbi.jacobian_health(J, x, zero.z_star)
        use_ls = False
        was_triggered = False
        ls_info = None

        if row["kind"] == "hybrid":
            F0 = np.asarray(F(z_zero, x), dtype=np.float64)
            F0_norm = float(np.linalg.norm(F0))
            if hybrid_cfg.use_relative_residual_trigger:
                tau_eff = max(
                    float(hybrid_cfg.tau_r),
                    float(hybrid_cfg.residual_trigger_scale) * F0_norm,
                )
            else:
                tau_eff = float(hybrid_cfg.tau_r)

            reasons = []
            if zero.residual > tau_eff:
                reasons.append("residual")
            if s_zero < hybrid_cfg.tau_sigma:
                reasons.append("conditioning")
            if hybrid_cfg.trigger_on_solver_failure and not zero.solver_success_flag:
                reasons.append("solver_failure")
            if hybrid_cfg.trigger_on_bad_jacobian and not jac_zero_ok:
                reasons.append("bad_jacobian")
            was_triggered = bool(reasons)

            if was_triggered:
                z_ls, ls_info = build_ls_candidate(F, J, x, layer.d, radius)
                det = mrbi.solve_root(F, J, x, z_ls, cfg=root_cfg)
                s_det, jac_det_ok = mrbi.jacobian_health(J, x, det.z_star)

                if (
                    hybrid_cfg.accept_any_finite_success_if_zero_failed
                    and not zero.solver_success_flag
                    and det.success
                    and np.isfinite(det.residual)
                ):
                    use_ls = True
                else:
                    accept_limit = float(hybrid_cfg.accept_factor_vs_zero) * zero.residual
                    competitive = bool(det.residual <= accept_limit)
                    clearly_better = bool(
                        det.residual
                        < (1.0 - float(hybrid_cfg.tie_residual_rtol)) * zero.residual
                    )
                    if competitive:
                        if clearly_better:
                            use_ls = True
                        elif (
                            hybrid_cfg.prefer_better_conditioning_on_tie
                            and jac_det_ok
                            and s_det > s_zero
                        ):
                            use_ls = True
                final = det if use_ls else zero
            else:
                final = zero

        elif row["kind"] == "forced":
            was_triggered = True
            z_ls, ls_info = build_ls_candidate(F, J, x, layer.d, radius)
            det = mrbi.solve_root(F, J, x, z_ls, cfg=root_cfg)
            s_det, _ = mrbi.jacobian_health(J, x, det.z_star)
            use_ls = bool(
                (not zero.success and det.success)
                or (np.isfinite(det.residual) and det.residual < zero.residual)
                or (
                    np.isfinite(det.residual)
                    and np.isfinite(zero.residual)
                    and det.residual <= 1.10 * zero.residual
                    and s_det > s_zero
                )
            )
            final = det if use_ls else zero
        else:
            raise RuntimeError(f"Unknown kind {row['kind']}")

        dist = float(np.linalg.norm(final.z_star - zero.z_star))
        Z.append(final.z_star)
        success.append(int(final.success))
        trigger.append(int(was_triggered))
        accepted.append(int(use_ls))
        changed.append(int(dist > 1e-8))
        distances.append(dist)
        residuals.append(float(final.residual))
        runtimes.append(float(time.perf_counter() - t0))
        if ls_info is not None:
            candidate_residuals.append(float(ls_info["residual"]))
            candidate_nfev.append(int(ls_info["nfev"]))

    return np.asarray(Z, dtype=np.float64), {
        "success_rate": float(np.mean(success)),
        "trigger_rate": float(np.mean(trigger)),
        "accept_rate": float(np.mean(accepted)),
        "changed_rate": float(np.mean(changed)),
        "mean_distance_from_zero": float(np.mean(distances)),
        "median_distance_from_zero": float(np.median(distances)),
        "mean_residual": float(np.mean(residuals)),
        "mean_candidate_residual": (
            float(np.mean(candidate_residuals)) if candidate_residuals else None
        ),
        "median_candidate_residual": (
            float(np.median(candidate_residuals)) if candidate_residuals else None
        ),
        "mean_candidate_nfev": float(np.mean(candidate_nfev)) if candidate_nfev else 0.0,
        "mean_runtime_sec_per_sample": float(np.mean(runtimes)),
    }


def run_one(dataset, seed, lock):
    cfg = base.make_cfg(seed, lock)
    row, cfg_sel = base.selected_cfg(dataset, cfg, lock)
    X, y = bench.load_dataset(dataset, cfg)
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )
    Xtr_p, Xte_p = bench.preprocess_to_pca(Xtr_raw, Xte_raw, cfg.pca_dim, cfg.seed)
    layer_seed = cfg.seed + 1000 + cfg.latent_dim + int(100 * cfg.spectral_radius)
    layer = bench.make_random_implicit_layer(
        dx=Xtr_p.shape[1], d=cfg.latent_dim, cfg=cfg, seed=layer_seed
    )
    Ztr, st_tr = solve_ls_policy(Xtr_p, layer, cfg_sel, row)
    Zte, st_te = solve_ls_policy(Xte_p, layer, cfg_sel, row)
    Ztr, Zte = bench.standardize_pair(Ztr, Zte)
    metrics = base.qnn_metrics(Ztr, ytr, Zte, yte, cfg)
    return {
        "dataset": dataset,
        "seed": seed,
        "selected_method": row["method"],
        "kind": row["kind"],
        "profile": row["profile"],
        "hybrid": row["hybrid"],
        "search_radius": float(cfg_sel.search_radius),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "solver_train": st_tr,
        "solver_test": st_te,
    }


def load_baseline():
    with BASELINE.open(newline="", encoding="utf-8") as f:
        return {row["dataset"]: row for row in csv.DictReader(f)}


def summarize(rows, datasets):
    baseline = load_baseline()
    summary_rows = []
    deltas = []
    for dataset in datasets:
        rr = [r for r in rows if r["dataset"] == dataset]
        ba = float(np.mean([r["balanced_accuracy"] for r in rr]))
        success = float(np.mean([r["solver_test"]["success_rate"] for r in rr]))
        accept = float(np.mean([r["solver_test"]["accept_rate"] for r in rr]))
        changed = float(np.mean([r["solver_test"]["changed_rate"] for r in rr]))
        runtime = float(
            np.mean([r["solver_test"]["mean_runtime_sec_per_sample"] for r in rr])
        )
        candidate_resid = [
            r["solver_test"]["mean_candidate_residual"]
            for r in rr
            if r["solver_test"]["mean_candidate_residual"] is not None
        ]
        mrbi_ba = float(baseline[dataset]["selected_mrbi"])
        zero_ba = float(baseline[dataset]["zero"])
        ms_ba = float(baseline[dataset]["multistart5"])
        delta = mrbi_ba - ba
        deltas.append(delta)
        summary_rows.append(
            {
                "dataset": dataset,
                "zero_ba": zero_ba,
                "multistart5_ba": ms_ba,
                "hybrid_ls_ba": ba,
                "selected_mrbi_ba": mrbi_ba,
                "delta_mrbi_minus_hybrid_ls": delta,
                "hybrid_ls_success": success,
                "hybrid_ls_accept_rate": accept,
                "hybrid_ls_changed_rate": changed,
                "hybrid_ls_mean_candidate_residual": (
                    float(np.mean(candidate_resid)) if candidate_resid else None
                ),
                "hybrid_ls_runtime_sec_per_sample": runtime,
            }
        )

    statistic, pvalue = exact_signed_rank_greater(deltas)
    ci = bootstrap_ci(deltas)
    arr = np.asarray(deltas, dtype=float)
    overall = {
        "n_datasets": len(datasets),
        "ls_max_nfev": LS_MAX_NFEV,
        "hybrid_ls_ba_mean": float(np.mean([r["hybrid_ls_ba"] for r in summary_rows])),
        "selected_mrbi_ba_mean": float(
            np.mean([r["selected_mrbi_ba"] for r in summary_rows])
        ),
        "delta_mrbi_minus_hybrid_ls_mean": float(np.mean(arr)),
        "delta_mrbi_minus_hybrid_ls_median": float(np.median(arr)),
        "delta_95pct_dataset_bootstrap": ci,
        "positive_datasets": int(np.sum(arr > 1e-12)),
        "neutral_datasets": int(np.sum(np.abs(arr) <= 1e-12)),
        "negative_datasets": int(np.sum(arr < -1e-12)),
        "exact_signed_rank_statistic": statistic,
        "exact_signed_rank_p_greater": pvalue,
        "hybrid_ls_success_mean": float(
            np.mean([r["hybrid_ls_success"] for r in summary_rows])
        ),
        "hybrid_ls_accept_rate_mean": float(
            np.mean([r["hybrid_ls_accept_rate"] for r in summary_rows])
        ),
        "hybrid_ls_changed_rate_mean": float(
            np.mean([r["hybrid_ls_changed_rate"] for r in summary_rows])
        ),
    }
    return summary_rows, overall


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = base.load_lock()
    datasets = tuple(lock["development_evidence"]["datasets"])
    seeds = tuple(lock["confirmation_plan"]["seeds"])
    checkpoint = OUT / "checkpoint.json"
    rows = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else []
    completed = {(r["dataset"], int(r["seed"])) for r in rows}

    for seed in seeds:
        for dataset in datasets:
            key = (dataset, seed)
            if key in completed:
                print(f"SKIP dataset={dataset} seed={seed}", flush=True)
                continue
            print(f"HYBRID_LS_START dataset={dataset} seed={seed}", flush=True)
            result = run_one(dataset, seed, lock)
            rows.append(result)
            checkpoint.write_text(
                json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            print(
                "HYBRID_LS_DONE "
                f"dataset={dataset} seed={seed} "
                f"ba={result['balanced_accuracy']:.4f} "
                f"success={result['solver_test']['success_rate']:.4f} "
                f"accept={result['solver_test']['accept_rate']:.4f} "
                f"changed={result['solver_test']['changed_rate']:.4f}",
                flush=True,
            )

    expected = len(datasets) * len(seeds)
    if len(rows) != expected:
        raise RuntimeError(f"Incomplete control run {len(rows)}/{expected}")

    summary_rows, overall = summarize(rows, datasets)
    csv_path = OUT / "dataset_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_path = OUT / "summary.json"
    summary_path.write_text(
        json.dumps(
            {"plan": PLAN, "dataset_summary": summary_rows, "overall": overall},
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("HYBRID_LEAST_SQUARES_CONTROL_OK")
    print(f"MRBI mean BA = {overall['selected_mrbi_ba_mean']:.6f}")
    print(f"Hybrid-LS mean BA = {overall['hybrid_ls_ba_mean']:.6f}")
    print(f"MRBI - Hybrid-LS = {overall['delta_mrbi_minus_hybrid_ls_mean']:+.6f}")
    print(f"95% dataset bootstrap = {overall['delta_95pct_dataset_bootstrap']}")
    print(f"Exact one-sided signed-rank p = {overall['exact_signed_rank_p_greater']:.6f}")
    print(
        "Dataset signs +/0/- = "
        f"{overall['positive_datasets']}/{overall['neutral_datasets']}/{overall['negative_datasets']}"
    )
    print(f"Hybrid-LS success = {overall['hybrid_ls_success_mean']:.6f}")
    print(f"SUMMARY {summary_path}")
    print(f"DATASET_TABLE {csv_path}")


if __name__ == "__main__":
    main()
