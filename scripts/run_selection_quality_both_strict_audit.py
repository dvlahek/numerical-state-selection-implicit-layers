#!/usr/bin/env python3
"""Build method-specific strict/different-state masks and classical readout summaries.

The script reuses the fixed feature cache and replays test numerical paths only
to recover per-sample strict-success flags that were not stored originally.
No QNN is trained and no experimental condition is changed.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

import main_qnn_benchmark as bench
import mrbi
import run_locked_selected_profile_confirmation as confirm
import run_hybrid_random_control as random_control
import run_least_squares_control as ls_control
import run_qnn_multiseed_all_paths as allpaths
import run_all_path_classical_readouts_rho2 as classical

PLAN = "selection_quality_both_strict_v1"
OUT = ROOT / "outputs" / PLAN
FEATURE_ROOT = ROOT / "outputs" / "qnn_multiseed_all_paths_v1" / "features"
SELECTOR_ROWS = ROOT / "outputs" / "all_path_rho_selector_sweep_v1" / "sample_rows.csv"

RHO = 2.0
STATE_TOL = 1e-6
CHECK_TOL = 5e-9
METHODS = ("selected_mrbi", "hybrid_ls", "hybrid_random", "multistart5")
READOUTS = tuple(classical.READOUTS)


def rel_distance(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(
        np.linalg.norm(a - b)
        / (1.0 + np.linalg.norm(a) + np.linalg.norm(b))
    )


def make_layer(cfg, Xtr_pca):
    seed = cfg.seed + 1000 + cfg.latent_dim + int(100 * cfg.spectral_radius)
    return bench.make_random_implicit_layer(
        dx=Xtr_pca.shape[1],
        d=cfg.latent_dim,
        cfg=cfg,
        seed=seed,
    )


def replay_zero(X, layer, cfg):
    F, J = mrbi.make_residual_and_jacobian(layer)
    root_cfg = bench.make_root_cfg(cfg)
    z0 = np.zeros(layer.d, dtype=np.float64)
    states, success = [], []
    for x in np.asarray(X, dtype=np.float64):
        r = mrbi.solve_root(F, J, x, z0, cfg=root_cfg)
        states.append(r.z_star)
        success.append(int(r.success))
    return np.asarray(states, dtype=np.float64), np.asarray(success, dtype=int)


def replay_hybrid_mrbi(X, layer, cfg, rng_seed):
    root_cfg = bench.make_root_cfg(cfg)
    mrbi_cfg = bench.make_mrbi_cfg(cfg)
    hybrid_cfg = bench.make_hybrid_cfg(cfg)
    rng = np.random.default_rng(rng_seed)

    states, success = [], []
    for x in np.asarray(X, dtype=np.float64):
        res = mrbi.hybrid_mrbi_solve_tanh_layer(
            layer,
            x,
            mrbi_cfg=mrbi_cfg,
            hybrid_cfg=hybrid_cfg,
            root_cfg=root_cfg,
            rng=rng,
        )
        states.append(res.z_star)
        success.append(int(res.success))
    return np.asarray(states, dtype=np.float64), np.asarray(success, dtype=int)


def replay_forced_mrbi(X, layer, cfg, rng_seed):
    F, J = mrbi.make_residual_and_jacobian(layer)
    root_cfg = bench.make_root_cfg(cfg)
    mrbi_cfg = bench.make_mrbi_cfg(cfg)
    rng = np.random.default_rng(rng_seed)
    z0 = np.zeros(layer.d, dtype=np.float64)

    states, success = [], []
    for x in np.asarray(X, dtype=np.float64):
        zero = mrbi.solve_root(F, J, x, z0, cfg=root_cfg)
        s0, _ = mrbi.jacobian_health(J, x, zero.z_star)

        opt = mrbi.MRBIOptimizer(
            F=F,
            J=J,
            x=x,
            dim=layer.d,
            config=mrbi_cfg,
            rng=rng,
        )
        cand = opt.optimize()
        det = mrbi.solve_root(F, J, x, cand.z_init, cfg=root_cfg)
        sd, _ = mrbi.jacobian_health(J, x, det.z_star)

        if cfg.forced_accept_if_better_residual:
            use_mrbi = bool(
                (not zero.success and det.success)
                or (np.isfinite(det.residual) and det.residual < zero.residual)
                or (
                    np.isfinite(det.residual)
                    and np.isfinite(zero.residual)
                    and det.residual <= 1.10 * zero.residual
                    and sd > s0
                )
            )
        else:
            use_mrbi = True

        final = det if use_mrbi else zero
        states.append(final.z_star)
        success.append(int(final.success))

    return np.asarray(states, dtype=np.float64), np.asarray(success, dtype=int)


def replay_selected_mrbi(X, layer, cfg_sel, row, data_seed):
    if row["kind"] == "forced":
        return replay_forced_mrbi(X, layer, cfg_sel, data_seed + 3000)
    if row["kind"] == "hybrid":
        return replay_hybrid_mrbi(X, layer, cfg_sel, data_seed + 5000)
    raise RuntimeError(f"Unknown selected kind: {row['kind']}")


def replay_random(X, layer, cfg, row, rng_seed):
    F, J = mrbi.make_residual_and_jacobian(layer)
    root_cfg = bench.make_root_cfg(cfg)
    rng = np.random.default_rng(rng_seed)
    hybrid_cfg = bench.make_hybrid_cfg(cfg) if row["kind"] == "hybrid" else None
    z_zero = np.zeros(layer.d, dtype=np.float64)
    radius = float(cfg.search_radius)

    states, success = [], []

    for x in np.asarray(X, dtype=np.float64):
        zero = mrbi.solve_root(F, J, x, z_zero, cfg=root_cfg)
        s_zero, jac_zero_ok = mrbi.jacobian_health(J, x, zero.z_star)
        use_random = False

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

            if reasons:
                z_random = rng.uniform(-radius, radius, size=layer.d)
                rnd = mrbi.solve_root(F, J, x, z_random, cfg=root_cfg)
                s_rnd, jac_rnd_ok = mrbi.jacobian_health(J, x, rnd.z_star)

                if (
                    hybrid_cfg.accept_any_finite_success_if_zero_failed
                    and not zero.solver_success_flag
                    and rnd.success
                    and np.isfinite(rnd.residual)
                ):
                    use_random = True
                else:
                    accept_limit = float(hybrid_cfg.accept_factor_vs_zero) * zero.residual
                    competitive = bool(rnd.residual <= accept_limit)
                    clearly_better = bool(
                        rnd.residual
                        < (1.0 - float(hybrid_cfg.tie_residual_rtol)) * zero.residual
                    )
                    if competitive:
                        if clearly_better:
                            use_random = True
                        elif (
                            hybrid_cfg.prefer_better_conditioning_on_tie
                            and jac_rnd_ok
                            and s_rnd > s_zero
                        ):
                            use_random = True
                final = rnd if use_random else zero
            else:
                final = zero

        elif row["kind"] == "forced":
            z_random = rng.uniform(-radius, radius, size=layer.d)
            rnd = mrbi.solve_root(F, J, x, z_random, cfg=root_cfg)
            s_rnd, _ = mrbi.jacobian_health(J, x, rnd.z_star)
            use_random = bool(
                (not zero.success and rnd.success)
                or (np.isfinite(rnd.residual) and rnd.residual < zero.residual)
                or (
                    np.isfinite(rnd.residual)
                    and np.isfinite(zero.residual)
                    and rnd.residual <= 1.10 * zero.residual
                    and s_rnd > s_zero
                )
            )
            final = rnd if use_random else zero
        else:
            raise RuntimeError(f"Unknown kind: {row['kind']}")

        states.append(final.z_star)
        success.append(int(final.success))

    return np.asarray(states, dtype=np.float64), np.asarray(success, dtype=int)


def replay_ls(X, layer, cfg, row):
    F, J = mrbi.make_residual_and_jacobian(layer)
    root_cfg = bench.make_root_cfg(cfg)
    hybrid_cfg = bench.make_hybrid_cfg(cfg) if row["kind"] == "hybrid" else None
    z_zero = np.zeros(layer.d, dtype=np.float64)
    radius = float(cfg.search_radius)

    states, success = [], []

    for x in np.asarray(X, dtype=np.float64):
        zero = mrbi.solve_root(F, J, x, z_zero, cfg=root_cfg)
        s_zero, jac_zero_ok = mrbi.jacobian_health(J, x, zero.z_star)
        use_ls = False

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

            if reasons:
                z_ls, _ = ls_control.build_ls_candidate(F, J, x, layer.d, radius)
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
            z_ls, _ = ls_control.build_ls_candidate(F, J, x, layer.d, radius)
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
            raise RuntimeError(f"Unknown kind: {row['kind']}")

        states.append(final.z_star)
        success.append(int(final.success))

    return np.asarray(states, dtype=np.float64), np.asarray(success, dtype=int)


def aggregate_equal_dataset(dataset_seed_df, value_col):
    d = dataset_seed_df[np.isfinite(dataset_seed_df[value_col].astype(float))].copy()
    if d.empty:
        return None, {}, 0
    per_dataset = d.groupby("dataset", sort=True)[value_col].mean()
    return (
        float(per_dataset.mean()),
        {str(k): float(v) for k, v in per_dataset.items()},
        int(len(per_dataset)),
    )


def main():
    if not SELECTOR_ROWS.exists():
        raise SystemExit(f"Missing selector rows: {SELECTOR_ROWS}")

    OUT.mkdir(parents=True, exist_ok=True)
    lock = confirm.load_lock()
    datasets = list(lock["development_evidence"]["datasets"])
    seeds = list(lock["confirmation_plan"]["seeds"])

    selector_all = pd.read_csv(SELECTOR_ROWS)
    selector = selector_all[
        np.isclose(selector_all["rho"].astype(float), RHO, atol=1e-12)
    ].copy()
    if selector.empty:
        raise SystemExit("No rho=2.0 rows in selector sample cache")

    required_selector = {
        "dataset", "seed", "position", "label",
        "zero_success_from_multistart_probe",
        "multistart_selected_success",
        "rel_distance_selected_mrbi_vs_zero",
        "rel_distance_hybrid_ls_vs_zero",
        "rel_distance_hybrid_random_vs_zero",
        "rel_distance_multistart5_vs_zero",
    }
    missing = sorted(required_selector - set(selector.columns))
    if missing:
        raise SystemExit(f"Selector cache missing columns: {missing}")

    case_rows = []
    ds_seed_rows = []

    for dataset_index, dataset in enumerate(datasets):
        for seed in seeds:
            print(f"AUDIT_START dataset={dataset} seed={seed}", flush=True)

            feature_path = FEATURE_ROOT / f"seed{seed}" / f"{dataset}.npz"
            if not feature_path.exists():
                raise SystemExit(f"Missing feature cache: {feature_path}")

            with np.load(feature_path, allow_pickle=False) as raw:
                if "Xtr_pca" not in raw.files or "Xte_pca" not in raw.files:
                    raise RuntimeError(f"PCA arrays missing from {feature_path}")
                Xtr_pca = np.asarray(raw["Xtr_pca"], dtype=np.float64)
                Xte_pca = np.asarray(raw["Xte_pca"], dtype=np.float64)

            ytr, yte, reps = allpaths.load_feature_cache(feature_path)
            ytr = np.asarray(ytr, dtype=int)
            yte = np.asarray(yte, dtype=int)

            g = selector[
                (selector["dataset"] == dataset)
                & (selector["seed"].astype(int) == int(seed))
            ].sort_values("position").reset_index(drop=True)

            if len(g) != len(yte):
                raise RuntimeError(
                    f"Length mismatch {dataset} seed={seed}: selector={len(g)} yte={len(yte)}"
                )
            if not np.array_equal(g["position"].to_numpy(int), np.arange(len(yte))):
                raise RuntimeError(f"Position mismatch {dataset} seed={seed}")
            if not np.array_equal(g["label"].to_numpy(int), yte):
                raise RuntimeError(f"Label mismatch {dataset} seed={seed}")

            cfg = confirm.make_cfg(seed, lock)
            row, cfg_sel = confirm.selected_cfg(dataset, cfg, lock)
            layer = make_layer(cfg, Xtr_pca)

            Z_zero, succ_zero_replay = replay_zero(Xte_pca, layer, cfg)
            Z_mrbi, succ_mrbi = replay_selected_mrbi(
                Xte_pca, layer, cfg_sel, row, seed
            )
            Z_ls, succ_ls = replay_ls(Xte_pca, layer, cfg_sel, row)
            Z_random, succ_random = replay_random(
                Xte_pca,
                layer,
                cfg_sel,
                row,
                seed + 9000 + 100 * dataset_index,
            )

            succ_zero_cached = g["zero_success_from_multistart_probe"].to_numpy(int)
            if not np.array_equal(succ_zero_replay, succ_zero_cached):
                bad = int(np.sum(succ_zero_replay != succ_zero_cached))
                raise RuntimeError(
                    f"Zero strict-flag replay mismatch {dataset} seed={seed}: {bad}"
                )

            replay_states = {
                "selected_mrbi": Z_mrbi,
                "hybrid_ls": Z_ls,
                "hybrid_random": Z_random,
            }
            replay_success = {
                "selected_mrbi": succ_mrbi,
                "hybrid_ls": succ_ls,
                "hybrid_random": succ_random,
                "multistart5": g["multistart_selected_success"].to_numpy(int),
            }
            distance_cols = {
                "selected_mrbi": "rel_distance_selected_mrbi_vs_zero",
                "hybrid_ls": "rel_distance_hybrid_ls_vs_zero",
                "hybrid_random": "rel_distance_hybrid_random_vs_zero",
                "multistart5": "rel_distance_multistart5_vs_zero",
            }

            for method in ("selected_mrbi", "hybrid_ls", "hybrid_random"):
                d_replay = np.asarray([
                    rel_distance(a, b)
                    for a, b in zip(replay_states[method], Z_zero)
                ])
                d_cached = g[distance_cols[method]].to_numpy(float)
                maxerr = float(np.max(np.abs(d_replay - d_cached)))
                if maxerr > CHECK_TOL:
                    raise RuntimeError(
                        f"Replay distance mismatch {dataset} seed={seed} "
                        f"method={method} max_abs_error={maxerr:.3e}"
                    )

            preds = {}
            correct = {}
            for readout in READOUTS:
                preds[readout] = {}
                correct[readout] = {}
                for method in ("zero",) + METHODS:
                    Xtr, Xte = reps[method]
                    model = classical.make_model(readout, seed)
                    model.fit(Xtr, ytr)
                    pred = np.asarray(model.predict(Xte), dtype=int)
                    preds[readout][method] = pred
                    correct[readout][method] = (pred == yte).astype(int)

            zero_strict = succ_zero_cached.astype(bool)

            for i in range(len(yte)):
                cr = {
                    "dataset": dataset,
                    "seed": int(seed),
                    "position": int(i),
                    "label": int(yte[i]),
                    "zero_strict": int(zero_strict[i]),
                }
                for method in METHODS:
                    method_strict = bool(replay_success[method][i])
                    d = float(g.loc[i, distance_cols[method]])
                    changed = bool(d > STATE_TOL)
                    mask = bool(zero_strict[i] and method_strict and changed)
                    cr[f"{method}_strict"] = int(method_strict)
                    cr[f"{method}_rel_distance_vs_zero"] = d
                    cr[f"{method}_different_state"] = int(changed)
                    cr[f"{method}_both_strict_different"] = int(mask)
                case_rows.append(cr)

            for method in METHODS:
                mask = (
                    zero_strict
                    & replay_success[method].astype(bool)
                    & (g[distance_cols[method]].to_numpy(float) > STATE_TOL)
                )
                n = int(np.sum(mask))

                for readout in READOUTS:
                    if n:
                        c0 = correct[readout]["zero"][mask]
                        cm = correct[readout][method][mask]
                        p0 = preds[readout]["zero"][mask]
                        pm = preds[readout][method][mask]
                        delta = float(np.mean(cm - c0))
                        disagree = float(np.mean(pm != p0))
                        acc0 = float(np.mean(c0))
                        accm = float(np.mean(cm))
                    else:
                        delta = np.nan
                        disagree = np.nan
                        acc0 = np.nan
                        accm = np.nan

                    ds_seed_rows.append({
                        "dataset": dataset,
                        "seed": int(seed),
                        "method": method,
                        "readout": readout,
                        "n_both_strict_different": n,
                        "zero_accuracy_on_subset": acc0,
                        "method_accuracy_on_subset": accm,
                        "accuracy_delta_method_minus_zero": delta,
                        "prediction_disagreement": disagree,
                    })

            print(
                "AUDIT_DONE "
                f"dataset={dataset} seed={seed} "
                + " ".join(
                    f"{m}={int(np.sum(zero_strict & replay_success[m].astype(bool) & (g[distance_cols[m]].to_numpy(float) > STATE_TOL)))}"
                    for m in METHODS
                ),
                flush=True,
            )

    case_df = pd.DataFrame(case_rows)
    dss_df = pd.DataFrame(ds_seed_rows)
    case_df.to_csv(OUT / "case_masks.csv", index=False)
    dss_df.to_csv(OUT / "dataset_seed_readout_summary.csv", index=False)

    dataset_rows = []
    overall_rows = []
    summary = {
        "plan": PLAN,
        "analysis_role": "post-confirmation selection-quality diagnostic",
        "rho": RHO,
        "state_match_threshold": STATE_TOL,
        "methods": list(METHODS),
        "readouts": list(READOUTS),
        "notes": [
            "Classical readouts use the existing cached feature matrices.",
            "Test numerical paths are replayed only to recover per-sample strict-success flags omitted from the original cache.",
            "Each method uses its own both-strict/different-state subset against Zero; cross-method subset sizes therefore differ.",
            "Results are descriptive and post-confirmation.",
        ],
        "results": {},
    }

    for method in METHODS:
        summary["results"][method] = {}
        for readout in READOUTS:
            d = dss_df[
                (dss_df["method"] == method)
                & (dss_df["readout"] == readout)
            ].copy()

            per_dataset = []
            for dataset, gd in d.groupby("dataset", sort=True):
                valid = gd[np.isfinite(gd["accuracy_delta_method_minus_zero"])].copy()
                if valid.empty:
                    continue
                per_dataset.append({
                    "dataset": dataset,
                    "method": method,
                    "readout": readout,
                    "seed_cases_present": int(len(valid)),
                    "n_subset_samples_sum": int(valid["n_both_strict_different"].sum()),
                    "mean_accuracy_delta_method_minus_zero": float(
                        valid["accuracy_delta_method_minus_zero"].mean()
                    ),
                    "mean_prediction_disagreement": float(
                        valid["prediction_disagreement"].mean()
                    ),
                    "mean_zero_accuracy_on_subset": float(
                        valid["zero_accuracy_on_subset"].mean()
                    ),
                    "mean_method_accuracy_on_subset": float(
                        valid["method_accuracy_on_subset"].mean()
                    ),
                })
            dataset_rows.extend(per_dataset)
            pdd = pd.DataFrame(per_dataset)

            if pdd.empty:
                overall = {
                    "datasets_present": 0,
                    "dataset_equal_mean_accuracy_delta": None,
                    "dataset_equal_mean_prediction_disagreement": None,
                    "positive_neutral_negative_datasets": {
                        "positive": 0, "neutral": 0, "negative": 0
                    },
                    "n_subset_samples_sum": 0,
                }
            else:
                vals = pdd["mean_accuracy_delta_method_minus_zero"].to_numpy(float)
                overall = {
                    "datasets_present": int(len(pdd)),
                    "dataset_equal_mean_accuracy_delta": float(np.mean(vals)),
                    "dataset_equal_median_accuracy_delta": float(np.median(vals)),
                    "dataset_equal_mean_prediction_disagreement": float(
                        pdd["mean_prediction_disagreement"].mean()
                    ),
                    "positive_neutral_negative_datasets": {
                        "positive": int(np.sum(vals > 1e-12)),
                        "neutral": int(np.sum(np.abs(vals) <= 1e-12)),
                        "negative": int(np.sum(vals < -1e-12)),
                    },
                    "n_subset_samples_sum": int(pdd["n_subset_samples_sum"].sum()),
                    "dataset_deltas": {
                        r["dataset"]: r["mean_accuracy_delta_method_minus_zero"]
                        for r in per_dataset
                    },
                }

            summary["results"][method][readout] = overall
            overall_rows.append({
                "method": method,
                "readout": readout,
                **{
                    k: v for k, v in overall.items()
                    if k not in ("positive_neutral_negative_datasets", "dataset_deltas")
                },
                "positive_datasets": overall["positive_neutral_negative_datasets"]["positive"],
                "neutral_datasets": overall["positive_neutral_negative_datasets"]["neutral"],
                "negative_datasets": overall["positive_neutral_negative_datasets"]["negative"],
            })

    dataset_df = pd.DataFrame(dataset_rows)
    overall_df = pd.DataFrame(overall_rows)

    dataset_df.to_csv(OUT / "dataset_readout_summary.csv", index=False)
    overall_df.to_csv(OUT / "overall_summary.csv", index=False)
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nSELECTION_QUALITY_AUDIT_OK")
    print(OUT / "summary.json")
    print("\nOVERALL TASK-EQUAL RESULTS")
    print(overall_df.to_string(index=False))


if __name__ == "__main__":
    main()
