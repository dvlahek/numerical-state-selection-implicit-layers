#!/usr/bin/env python3
"""Post-confirmation audit of the both-strict selector result.

No implicit feature matrices are regenerated and no QNN is retrained.

Checks:
1) trigger-conditioned audit:
   method-specific both-strict/different-state masks are intersected with the
   locked selected-policy trigger. Forced selected profiles count as active.
2) common-mask audit:
   Zero and all four alternatives must be strict on the same sample, and at
   least one alternative state must differ from Zero.

Classical readouts are refit on the fixed cached features. Saved four-seed QNN
predictions are loaded from disk.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

import main_qnn_benchmark as bench
import mrbi
import run_locked_selected_profile_confirmation as confirm
import run_qnn_multiseed_all_paths as allpaths
import run_all_path_classical_readouts_rho2 as classical

PLAN = "selector_confound_audit_v1"
DEFAULT_OUT = ROOT / "outputs" / PLAN

MASKS = ROOT / "outputs" / "selection_quality_both_strict_v1" / "case_masks.csv"
FEATURE_ROOT = ROOT / "outputs" / "qnn_multiseed_all_paths_v1" / "features"
QNN_PRED_ROOT = ROOT / "outputs" / "selection_quality_qnn_both_strict_v1" / "predictions"

METHODS = ("selected_mrbi", "hybrid_ls", "hybrid_random", "multistart5")
CLASSICAL_READOUTS = tuple(classical.READOUTS)
QNN_REPS = 4


def qnn_pred_path(dataset, seed, rep, method):
    return QNN_PRED_ROOT / f"{dataset}__seed{seed}__rep{rep}__{method}.npz"


def load_qnn_pred(dataset, seed, rep, method):
    p = qnn_pred_path(dataset, seed, rep, method)
    if not p.exists():
        raise SystemExit(
            f"Missing saved QNN prediction: {p}\n"
            "Run scripts/run_qnn_both_strict_selection_quality.py first. "
            "This audit itself does not retrain QNNs."
        )
    with np.load(p, allow_pickle=False) as raw:
        return np.asarray(raw["pred"], dtype=int)


def make_layer(cfg, Xtr_pca):
    layer_seed = cfg.seed + 1000 + cfg.latent_dim + int(100 * cfg.spectral_radius)
    return bench.make_random_implicit_layer(
        dx=Xtr_pca.shape[1],
        d=cfg.latent_dim,
        cfg=cfg,
        seed=layer_seed,
    )


def selected_policy_trigger_flags(Xte_pca, layer, cfg_sel, selected_row):
    """Replay only Zero-path quantities required by the locked trigger."""
    n = len(Xte_pca)

    if selected_row["kind"] == "forced":
        meta = pd.DataFrame(
            {
                "position": np.arange(n, dtype=int),
                "policy_triggered": np.ones(n, dtype=int),
                "trigger_residual": np.zeros(n, dtype=int),
                "trigger_conditioning": np.zeros(n, dtype=int),
                "trigger_solver_failure": np.zeros(n, dtype=int),
                "trigger_bad_jacobian": np.zeros(n, dtype=int),
                "forced_profile": np.ones(n, dtype=int),
            }
        )
        return np.ones(n, dtype=bool), meta

    if selected_row["kind"] != "hybrid":
        raise RuntimeError(f"Unknown selected kind: {selected_row['kind']}")

    F, J = mrbi.make_residual_and_jacobian(layer)
    root_cfg = bench.make_root_cfg(cfg_sel)
    hybrid_cfg = bench.make_hybrid_cfg(cfg_sel)
    z0 = np.zeros(layer.d, dtype=np.float64)

    flags = []
    rows = []
    for i, x in enumerate(np.asarray(Xte_pca, dtype=np.float64)):
        zero = mrbi.solve_root(F, J, x, z0, cfg=root_cfg)
        s0, jac_ok = mrbi.jacobian_health(J, x, zero.z_star)

        F0_norm = float(np.linalg.norm(np.asarray(F(z0, x), dtype=np.float64)))
        if hybrid_cfg.use_relative_residual_trigger:
            tau_eff = max(
                float(hybrid_cfg.tau_r),
                float(hybrid_cfg.residual_trigger_scale) * F0_norm,
            )
        else:
            tau_eff = float(hybrid_cfg.tau_r)

        r_resid = bool(zero.residual > tau_eff)
        r_cond = bool(s0 < hybrid_cfg.tau_sigma)
        r_fail = bool(
            hybrid_cfg.trigger_on_solver_failure and not zero.solver_success_flag
        )
        r_jac = bool(hybrid_cfg.trigger_on_bad_jacobian and not jac_ok)
        triggered = bool(r_resid or r_cond or r_fail or r_jac)

        flags.append(triggered)
        rows.append(
            {
                "position": int(i),
                "policy_triggered": int(triggered),
                "trigger_residual": int(r_resid),
                "trigger_conditioning": int(r_cond),
                "trigger_solver_failure": int(r_fail),
                "trigger_bad_jacobian": int(r_jac),
                "forced_profile": 0,
                "zero_residual_for_trigger": float(zero.residual),
                "zero_sigma_min_for_trigger": float(s0),
                "tau_r_eff": float(tau_eff),
                "tau_sigma": float(hybrid_cfg.tau_sigma),
            }
        )
    return np.asarray(flags, dtype=bool), pd.DataFrame(rows)


def sample_accuracy_delta(y, p0, pm, mask):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return np.nan, np.nan, 0
    y = np.asarray(y, dtype=int)
    p0 = np.asarray(p0, dtype=int)
    pm = np.asarray(pm, dtype=int)
    c0 = (p0[mask] == y[mask]).astype(float)
    cm = (pm[mask] == y[mask]).astype(float)
    return (
        float(np.mean(cm - c0)),
        float(np.mean(pm[mask] != p0[mask])),
        int(np.sum(mask)),
    )


def aggregate_task_equal(rows):
    out = []
    for (view, readout, method), g in rows.groupby(
        ["view", "readout", "method"], sort=False
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
                "view": view,
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if not args.run:
        raise SystemExit("Use --run")

    if not MASKS.exists():
        raise SystemExit(f"Missing masks: {MASKS}")
    if not QNN_PRED_ROOT.exists():
        raise SystemExit(f"Missing QNN predictions directory: {QNN_PRED_ROOT}")

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    masks = pd.read_csv(MASKS)
    lock = confirm.load_lock()
    datasets = list(lock["development_evidence"]["datasets"])
    seeds = list(lock["confirmation_plan"]["seeds"])

    case_rows = []
    result_rows = []

    for dataset in datasets:
        for seed in seeds:
            print(f"CONFOUND_AUDIT_START dataset={dataset} seed={seed}", flush=True)

            feature_path = FEATURE_ROOT / f"seed{seed}" / f"{dataset}.npz"
            if not feature_path.exists():
                raise SystemExit(f"Missing feature cache: {feature_path}")

            with np.load(feature_path, allow_pickle=False) as raw:
                Xtr_pca = np.asarray(raw["Xtr_pca"], dtype=np.float64)
                Xte_pca = np.asarray(raw["Xte_pca"], dtype=np.float64)

            ytr, yte, reps = allpaths.load_feature_cache(feature_path)
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
                required |= {
                    f"{method}_strict",
                    f"{method}_different_state",
                    f"{method}_both_strict_different",
                }
            missing = sorted(required - set(mg.columns))
            if missing:
                raise RuntimeError(f"Mask columns missing: {missing}")

            cfg = confirm.make_cfg(seed, lock)
            selected_row, cfg_sel = confirm.selected_cfg(dataset, cfg, lock)
            layer = make_layer(cfg, Xtr_pca)
            trigger, trigger_meta = selected_policy_trigger_flags(
                Xte_pca, layer, cfg_sel, selected_row
            )

            all_strict = mg["zero_strict"].to_numpy(int).astype(bool)
            any_changed = np.zeros(len(mg), dtype=bool)
            for method in METHODS:
                all_strict &= mg[f"{method}_strict"].to_numpy(int).astype(bool)
                any_changed |= (
                    mg[f"{method}_different_state"].to_numpy(int).astype(bool)
                )
            common_mask = all_strict & any_changed

            for i in range(len(mg)):
                row = {
                    "dataset": dataset,
                    "seed": int(seed),
                    "position": int(i),
                    "selected_kind": str(selected_row["kind"]),
                    "selected_profile": str(selected_row["profile"]),
                    "selected_hybrid": str(selected_row["hybrid"]),
                    "policy_triggered": int(trigger[i]),
                    "common_all_strict_any_changed": int(common_mask[i]),
                }
                for c in trigger_meta.columns:
                    if c == "position":
                        continue
                    val = trigger_meta.iloc[i][c]
                    if pd.notna(val):
                        if isinstance(val, (float, np.floating)):
                            row[c] = float(val)
                        else:
                            row[c] = int(val)
                case_rows.append(row)

            classical_pred = {}
            for readout in CLASSICAL_READOUTS:
                classical_pred[readout] = {}
                for method in ("zero",) + METHODS:
                    Xtr, Xte = reps[method]
                    model = classical.make_model(readout, seed)
                    model.fit(Xtr, ytr)
                    classical_pred[readout][method] = np.asarray(
                        model.predict(Xte), dtype=int
                    )

            for method in METHODS:
                original_mask = (
                    mg[f"{method}_both_strict_different"]
                    .to_numpy(int)
                    .astype(bool)
                )
                trigger_mask = original_mask & trigger

                for readout in CLASSICAL_READOUTS:
                    p0 = classical_pred[readout]["zero"]
                    pm = classical_pred[readout][method]
                    for view, mask in (
                        ("trigger_conditioned", trigger_mask),
                        ("common_all_strict", common_mask),
                    ):
                        delta, disagree, n = sample_accuracy_delta(
                            yte, p0, pm, mask
                        )
                        result_rows.append(
                            {
                                "dataset": dataset,
                                "seed": int(seed),
                                "qnn_rep": -1,
                                "readout": readout,
                                "method": method,
                                "view": view,
                                "n": n,
                                "accuracy_delta": delta,
                                "prediction_disagreement": disagree,
                            }
                        )

                for rep in range(QNN_REPS):
                    p0 = load_qnn_pred(dataset, seed, rep, "zero")
                    pm = load_qnn_pred(dataset, seed, rep, method)
                    for view, mask in (
                        ("trigger_conditioned", trigger_mask),
                        ("common_all_strict", common_mask),
                    ):
                        delta, disagree, n = sample_accuracy_delta(
                            yte, p0, pm, mask
                        )
                        result_rows.append(
                            {
                                "dataset": dataset,
                                "seed": int(seed),
                                "qnn_rep": int(rep),
                                "readout": "qnn",
                                "method": method,
                                "view": view,
                                "n": n,
                                "accuracy_delta": delta,
                                "prediction_disagreement": disagree,
                            }
                        )

            print(
                f"CONFOUND_AUDIT_DONE dataset={dataset} seed={seed} "
                f"trigger_rate={float(np.mean(trigger)):.3f} "
                f"common_n={int(np.sum(common_mask))}",
                flush=True,
            )

    cases = pd.DataFrame(case_rows)
    rows = pd.DataFrame(result_rows)
    overall = aggregate_task_equal(rows)

    cases.to_csv(out / "case_conditioning.csv", index=False)
    rows.to_csv(out / "dataset_seed_readout_summary.csv", index=False)
    overall.to_csv(out / "overall_summary.csv", index=False)
    overall[
        [
            "view",
            "readout",
            "method",
            "task_equal_accuracy_delta",
            "positive_tasks",
            "neutral_tasks",
            "negative_tasks",
            "n_tasks",
        ]
    ].to_csv(out / "compact_summary.csv", index=False)

    summary = {
        "plan": PLAN,
        "analysis_role": "post-confirmation descriptive confound audit",
        "uses_new_feature_generation": False,
        "uses_new_qnn_training": False,
        "trigger_definition": (
            "Locked selected-policy trigger evaluated from the Zero path; "
            "forced profiles count as always active."
        ),
        "trigger_conditioned_view": (
            "Method-specific both-strict/different-state mask intersected "
            "with the selected-policy trigger."
        ),
        "common_mask_view": (
            "Zero and all four alternatives strict, and at least one "
            "alternative returned state differs from Zero."
        ),
        "common_mask_note": (
            "All selectors are evaluated on the same test samples. "
            "Selectors returning the same state as Zero are not assigned "
            "zero utility manually; the established downstream prediction "
            "is used."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("SELECTOR_CONFOUND_AUDIT_OK", flush=True)
    print(out, flush=True)
    print("\nCOMPACT SUMMARY", flush=True)
    print(overall.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
