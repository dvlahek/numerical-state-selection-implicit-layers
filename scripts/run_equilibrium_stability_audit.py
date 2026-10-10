#!/usr/bin/env python3
"""Post-confirmation local fixed-point stability audit.

For each returned test state z, compute the spectral radius of the Jacobian of
the fixed-point map T(z)=tanh(Wz+Ux+b):

    rho_T(z) = rho(diag(1-z^2) W).

A returned strict root is locally attractive for the simple fixed-point
iteration when rho_T(z) < 1.

This audit replays only the test-set numerical paths to recover raw returned
states. It does not train QNNs or classical readouts and does not change any
development or confirmation decision.

Strict/non-strict labels and the common all-strict mask are taken from the
already established sample-level audit:
    outputs/selection_quality_both_strict_v1/case_masks.csv
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
import run_locked_selected_profile_confirmation as confirm
import run_hybrid_random_control as random_control
import run_least_squares_control as ls_control
import analyze_branch_consistency as branch

PLAN = "equilibrium_stability_audit_v1"
DEFAULT_OUT = ROOT / "outputs" / PLAN
CASE_MASKS = (
    ROOT
    / "outputs"
    / "selection_quality_both_strict_v1"
    / "case_masks.csv"
)

METHODS = (
    "zero",
    "selected_mrbi",
    "hybrid_ls",
    "hybrid_random",
    "multistart5",
)
ALT_METHODS = (
    "selected_mrbi",
    "hybrid_ls",
    "hybrid_random",
    "multistart5",
)
STABILITY_THRESHOLD = 1.0


def local_map_spectral_radius(z, W):
    """rho(diag(1-z^2) W) for the tanh fixed-point map."""
    z = np.asarray(z, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    J_map = (1.0 - np.square(z))[:, None] * W
    return float(np.max(np.abs(np.linalg.eigvals(J_map))))


def load_case_mask(dataset, seed, n_test):
    if not CASE_MASKS.exists():
        raise SystemExit(
            f"Missing established case mask: {CASE_MASKS}\n"
            "Run the existing both-strict audit first."
        )

    masks = pd.read_csv(CASE_MASKS)
    g = (
        masks[
            (masks["dataset"] == dataset)
            & (masks["seed"].astype(int) == int(seed))
        ]
        .sort_values("position")
        .reset_index(drop=True)
    )

    if len(g) != n_test:
        raise RuntimeError(
            f"Mask length mismatch for {dataset} seed={seed}: "
            f"{len(g)} != {n_test}"
        )

    pos = g["position"].to_numpy(int)
    if not np.array_equal(pos, np.arange(n_test)):
        raise RuntimeError(
            f"Mask positions are not 0..n_test-1 for {dataset} seed={seed}"
        )

    required = {"zero_strict"}
    for method in ALT_METHODS:
        required |= {
            f"{method}_strict",
            f"{method}_different_state",
        }
    missing = sorted(required - set(g.columns))
    if missing:
        raise RuntimeError(
            "case_masks.csv is missing required columns: " + ", ".join(missing)
        )

    return g


def replay_raw_states(dataset, seed, dataset_index, lock):
    """Replay the established rho=2.0 test-set numerical paths."""
    (
        cfg,
        selected_row,
        cfg_sel,
        Xtr_p,
        Xte_p,
        ytr,
        yte,
        layer,
    ) = branch.prepare(dataset, seed, lock)

    Z_zero, _ = bench.solve_zero_features(Xte_p, layer, cfg)

    Z_mrbi, _ = branch.solve_selected_test(
        Xte_p, layer, cfg_sel, selected_row, seed
    )

    Z_ls, _ = ls_control.solve_ls_policy(
        Xte_p, layer, cfg_sel, selected_row
    )

    Z_random, _ = random_control.solve_random_policy(
        Xte_p,
        layer,
        cfg_sel,
        selected_row,
        seed + 9000 + 100 * dataset_index,
    )

    n_starts = int(
        lock["confirmation_plan"]["random_multistart"]["n_random_starts"]
    )
    Z_ms, _ = branch.solve_multistart_detailed(
        Xte_p,
        layer,
        cfg,
        seed + 7000 + 100 * dataset_index,
        n_starts=n_starts,
        radius=1.0,
    )

    states = {
        "zero": np.asarray(Z_zero, dtype=np.float64),
        "selected_mrbi": np.asarray(Z_mrbi, dtype=np.float64),
        "hybrid_ls": np.asarray(Z_ls, dtype=np.float64),
        "hybrid_random": np.asarray(Z_random, dtype=np.float64),
        "multistart5": np.asarray(Z_ms, dtype=np.float64),
    }

    return cfg, selected_row, Xte_p, np.asarray(yte, dtype=int), layer, states


def task_equal_summary(sample_rows):
    df = pd.DataFrame(sample_rows)

    summaries = []
    for subset_name, subset_mask in (
        ("all_strict_returns", df["strict"].to_numpy(int) == 1),
        (
            "common_all_strict_any_changed",
            (df["strict"].to_numpy(int) == 1)
            & (df["common_all_strict_any_changed"].to_numpy(int) == 1),
        ),
    ):
        d = df.loc[subset_mask].copy()

        for method in METHODS:
            g = d[d["method"] == method].copy()
            if g.empty:
                continue

            seed_level = (
                g.groupby(["dataset", "seed"], as_index=False)
                .agg(
                    stable_fraction=("locally_attractive", "mean"),
                    mean_rho_map=("rho_map", "mean"),
                    median_rho_map=("rho_map", "median"),
                    n=("position", "count"),
                )
            )
            task_level = (
                seed_level.groupby("dataset", as_index=False)
                .agg(
                    stable_fraction=("stable_fraction", "mean"),
                    mean_rho_map=("mean_rho_map", "mean"),
                    median_rho_map=("median_rho_map", "mean"),
                    mean_n=("n", "mean"),
                )
            )

            summaries.append(
                {
                    "subset": subset_name,
                    "method": method,
                    "n_pooled_strict_states": int(len(g)),
                    "n_tasks": int(len(task_level)),
                    "task_equal_stable_fraction": float(
                        task_level["stable_fraction"].mean()
                    ),
                    "task_equal_mean_rho_map": float(
                        task_level["mean_rho_map"].mean()
                    ),
                    "task_equal_median_rho_map": float(
                        task_level["median_rho_map"].mean()
                    ),
                    "median_task_stable_fraction": float(
                        task_level["stable_fraction"].median()
                    ),
                }
            )

    return pd.DataFrame(summaries)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional single confirmation seed (40--49) for a smoke test.",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional single benchmark task for a smoke test.",
    )
    args = ap.parse_args()

    if not args.run:
        raise SystemExit("Use --run")

    lock = confirm.load_lock()
    all_datasets = list(lock["development_evidence"]["datasets"])
    all_seeds = list(lock["confirmation_plan"]["seeds"])

    datasets = all_datasets
    seeds = all_seeds

    if args.dataset is not None:
        if args.dataset not in all_datasets:
            raise SystemExit(
                f"Unknown dataset/task {args.dataset!r}; choose from {all_datasets}"
            )
        datasets = [args.dataset]

    if args.seed is not None:
        if args.seed not in all_seeds:
            raise SystemExit(
                f"Seed must be one of confirmation seeds {all_seeds}"
            )
        seeds = [args.seed]

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    sample_rows = []

    for seed in seeds:
        for dataset_index, dataset in enumerate(all_datasets):
            if dataset not in datasets:
                continue

            print(
                f"STABILITY_AUDIT_START dataset={dataset} seed={seed}",
                flush=True,
            )

            (
                cfg,
                selected_row,
                Xte_p,
                yte,
                layer,
                states,
            ) = replay_raw_states(
                dataset, seed, dataset_index, lock
            )

            n_test = len(yte)
            masks = load_case_mask(dataset, seed, n_test)

            all_strict = masks["zero_strict"].to_numpy(int).astype(bool)
            any_changed = np.zeros(n_test, dtype=bool)
            for method in ALT_METHODS:
                all_strict &= (
                    masks[f"{method}_strict"]
                    .to_numpy(int)
                    .astype(bool)
                )
                any_changed |= (
                    masks[f"{method}_different_state"]
                    .to_numpy(int)
                    .astype(bool)
                )
            common_mask = all_strict & any_changed

            strict_by_method = {
                "zero": masks["zero_strict"].to_numpy(int).astype(bool),
                **{
                    method: (
                        masks[f"{method}_strict"]
                        .to_numpy(int)
                        .astype(bool)
                    )
                    for method in ALT_METHODS
                },
            }

            W = np.asarray(layer.W, dtype=np.float64)

            for method in METHODS:
                Z = states[method]
                if len(Z) != n_test:
                    raise RuntimeError(
                        f"State length mismatch: {dataset} seed={seed} "
                        f"{method}: {len(Z)} != {n_test}"
                    )

                for i, z in enumerate(Z):
                    rho_map = local_map_spectral_radius(z, W)
                    sample_rows.append(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "position": int(i),
                            "label": int(yte[i]),
                            "method": method,
                            "strict": int(strict_by_method[method][i]),
                            "rho_map": rho_map,
                            "locally_attractive": int(
                                rho_map < STABILITY_THRESHOLD
                            ),
                            "common_all_strict_any_changed": int(
                                common_mask[i]
                            ),
                            "selected_profile": str(
                                selected_row["profile"]
                            ),
                            "selected_kind": str(
                                selected_row["kind"]
                            ),
                        }
                    )

            print(
                f"STABILITY_AUDIT_DONE dataset={dataset} seed={seed} "
                f"common_n={int(np.sum(common_mask))}",
                flush=True,
            )

    samples = pd.DataFrame(sample_rows)
    summary = task_equal_summary(sample_rows)

    samples.to_csv(out / "sample_rows.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)

    meta = {
        "plan": PLAN,
        "analysis_role": (
            "post-confirmation descriptive local fixed-point stability audit"
        ),
        "stability_measure": (
            "spectral radius of diag(1-z^2) W for T(z)=tanh(Wz+Ux+b)"
        ),
        "locally_attractive_threshold": "< 1",
        "uses_new_qnn_training": False,
        "uses_new_classical_training": False,
        "replays_test_numerical_paths": True,
        "strict_labels_source": str(CASE_MASKS.relative_to(ROOT)),
        "note": (
            "Current QNN feature caches contain standardized latent features, "
            "so raw test-set states are replayed before evaluating local "
            "fixed-point stability."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )

    print("EQUILIBRIUM_STABILITY_AUDIT_OK", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(out, flush=True)


if __name__ == "__main__":
    main()
