"""Multistart-vs-Zero state and logistic-prediction flip rates across rho.

Post-confirmation mechanistic diagnostic. Recomputes only Zero and Multistart-5.
No MRBI optimization and no QNN training.

Outputs to outputs/rho_multistart_prediction_flip_v1/.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

import main_qnn_benchmark as bench
import run_locked_selected_profile_confirmation as confirm
from fixed_profile_robustness_common import (
    RHO_VALUES, prepare_dataset, make_base_layer, layer_at_rho,
)

PLAN = "rho_multistart_prediction_flip_v1"
DEFAULT_OUT = ROOT / "outputs" / PLAN
MATCH_TOL = 1e-6


def rel_distance(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a-b) / (1.0 + np.linalg.norm(a) + np.linalg.norm(b)))


def make_lr(seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, random_state=seed),
    )


def solve_case(dataset, seed, dataset_index, rho, lock):
    cfg, Xtr_p, Xte_p, ytr, yte = prepare_dataset(dataset, seed, lock)
    base_layer, base_seed = make_base_layer(Xtr_p, cfg)
    layer, base_rho, actual_rho = layer_at_rho(base_layer, float(rho))

    Ztr_zero, _ = bench.solve_zero_features(Xtr_p, layer, cfg)
    Zte_zero, _ = bench.solve_zero_features(Xte_p, layer, cfg)

    n_starts = int(lock["confirmation_plan"]["random_multistart"]["n_random_starts"])
    Ztr_ms, _ = confirm.solve_multistart(
        Xtr_p, layer, cfg,
        rng_seed=seed + 6000 + 100 * dataset_index,
        n_starts=n_starts, radius=1.0,
    )
    Zte_ms, _ = confirm.solve_multistart(
        Xte_p, layer, cfg,
        rng_seed=seed + 7000 + 100 * dataset_index,
        n_starts=n_starts, radius=1.0,
    )

    lr_zero = make_lr(seed)
    lr_ms = make_lr(seed)
    lr_zero.fit(Ztr_zero, ytr)
    lr_ms.fit(Ztr_ms, ytr)
    pred_zero = np.asarray(lr_zero.predict(Zte_zero), dtype=int)
    pred_ms = np.asarray(lr_ms.predict(Zte_ms), dtype=int)

    distances = np.asarray([rel_distance(a, b) for a, b in zip(Zte_ms, Zte_zero)])
    state_changed = distances > MATCH_TOL
    pred_changed = pred_ms != pred_zero

    n_changed = int(np.sum(state_changed))
    joint = int(np.sum(pred_changed & state_changed))
    conditional = float(joint / n_changed) if n_changed else 0.0

    ba0 = float(balanced_accuracy_score(yte, pred_zero))
    bams = float(balanced_accuracy_score(yte, pred_ms))

    return {
        "dataset": dataset,
        "seed": int(seed),
        "rho": float(rho),
        "base_layer_seed": int(base_seed),
        "base_rho_actual": float(base_rho),
        "rho_actual": float(actual_rho),
        "n_test": int(len(yte)),
        "state_change_rate": float(np.mean(state_changed)),
        "prediction_flip_rate": float(np.mean(pred_changed)),
        "prediction_flip_given_state_change": conditional,
        "n_state_changed": n_changed,
        "n_prediction_flipped": int(np.sum(pred_changed)),
        "n_flip_and_state_changed": joint,
        "zero_ba": ba0,
        "multistart5_ba": bams,
        "ba_delta_multistart_minus_zero": bams - ba0,
    }


def aggregate(rows, datasets):
    out = {"by_rho": {}}
    for rho in sorted({float(r["rho"]) for r in rows}):
        rr = [r for r in rows if float(r["rho"]) == rho]
        ds_rows = []
        for dataset in datasets:
            dd = [r for r in rr if r["dataset"] == dataset]
            if not dd:
                continue
            n_changed = sum(r["n_state_changed"] for r in dd)
            n_joint = sum(r["n_flip_and_state_changed"] for r in dd)
            ds_rows.append({
                "dataset": dataset,
                "state_change_rate": float(np.mean([r["state_change_rate"] for r in dd])),
                "prediction_flip_rate": float(np.mean([r["prediction_flip_rate"] for r in dd])),
                "prediction_flip_given_state_change": float(n_joint/n_changed) if n_changed else 0.0,
                "zero_ba": float(np.mean([r["zero_ba"] for r in dd])),
                "multistart5_ba": float(np.mean([r["multistart5_ba"] for r in dd])),
            })
        out["by_rho"][f"{rho:.2f}"] = {
            "n_dataset_seed_cases": len(rr),
            "dataset_rows": ds_rows,
            "mean_dataset_state_change_rate": float(np.mean([d["state_change_rate"] for d in ds_rows])),
            "mean_dataset_prediction_flip_rate": float(np.mean([d["prediction_flip_rate"] for d in ds_rows])),
            "mean_dataset_prediction_flip_given_state_change": float(np.mean([d["prediction_flip_given_state_change"] for d in ds_rows])),
            "mean_dataset_zero_ba": float(np.mean([d["zero_ba"] for d in ds_rows])),
            "mean_dataset_multistart5_ba": float(np.mean([d["multistart5_ba"] for d in ds_rows])),
        }
    return out


def write_outputs(out_dir, rows, datasets):
    with (out_dir / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "plan": PLAN,
        "analysis_role": "post-confirmation mechanistic diagnostic",
        "state_match_threshold": MATCH_TOL,
        "aggregate": aggregate(rows, datasets),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    by = summary["aggregate"]["by_rho"]
    rhos = np.asarray(sorted(float(k) for k in by), dtype=float)
    state = np.asarray([by[f"{r:.2f}"]["mean_dataset_state_change_rate"] for r in rhos])
    pred = np.asarray([by[f"{r:.2f}"]["mean_dataset_prediction_flip_rate"] for r in rhos])

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(rhos, state, marker="o", label="Returned state differs")
    ax.plot(rhos, pred, marker="o", label="Logistic prediction differs")
    ax.set_xlabel(r"Spectral radius $\rho(W)$")
    ax.set_ylabel("Fraction of test samples")
    ax.set_xticks(rhos)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "rho_state_vs_logreg_flip.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "rho_state_vs_logreg_flip.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    lock = confirm.load_lock()
    datasets = list(lock["development_evidence"]["datasets"])
    seeds = list(lock["confirmation_plan"]["seeds"])
    rhos = [float(x) for x in RHO_VALUES]

    if args.seed is not None:
        if args.seed not in seeds:
            raise SystemExit("seed must be a locked confirmation seed")
        seeds = [args.seed]
    if args.rho is not None:
        if float(args.rho) not in rhos:
            raise SystemExit(f"rho must be one of {rhos}")
        rhos = [float(args.rho)]

    if args.dry_run:
        print(
            "RHO_PREDICTION_FLIP_DRY_RUN_OK "
            f"datasets={len(datasets)} seeds={seeds} rhos={rhos} "
            "paths=zero,multistart5 readout=logreg qnn=false",
            flush=True,
        )
        return

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "checkpoint.json"

    rows = []
    if checkpoint.exists():
        rows = json.loads(checkpoint.read_text(encoding="utf-8")).get("rows", [])

    completed = {(r["dataset"], int(r["seed"]), float(r["rho"])) for r in rows}
    done_now = 0

    for rho in rhos:
        for seed in seeds:
            for dataset_index, dataset in enumerate(datasets):
                key = (dataset, int(seed), float(rho))
                if key in completed:
                    print(f"SKIP dataset={dataset} seed={seed} rho={rho:.2f}", flush=True)
                    continue

                print(f"FLIP_START dataset={dataset} seed={seed} rho={rho:.2f}", flush=True)
                row = solve_case(dataset, seed, dataset_index, rho, lock)
                rows.append(row)
                completed.add(key)

                checkpoint.write_text(
                    json.dumps({"plan": PLAN, "rows": rows}, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )

                print(
                    "FLIP_DONE "
                    f"dataset={dataset} seed={seed} rho={rho:.2f} "
                    f"state={row['state_change_rate']:.4f} "
                    f"pred={row['prediction_flip_rate']:.4f} "
                    f"cond={row['prediction_flip_given_state_change']:.4f}",
                    flush=True,
                )

                done_now += 1
                if args.max_cases is not None and done_now >= args.max_cases:
                    write_outputs(out_dir, rows, datasets)
                    print("STOPPED_AT_MAX_CASES", flush=True)
                    return

    write_outputs(out_dir, rows, datasets)
    print(f"RHO_PREDICTION_FLIP_OK {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
