"""Post-confirmation all-path numerical selector sweep across fixed spectral radii.

Test-set only. No QNN training. Keeps the development-selected MRBI profile/policy
fixed and rescales only W through the existing fixed-profile robustness utilities.

Outputs:
  outputs/all_path_rho_selector_sweep_v1/checkpoint.json
  outputs/all_path_rho_selector_sweep_v1/summary.json
  outputs/all_path_rho_selector_sweep_v1/sample_rows.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

import main_qnn_benchmark as bench
import run_locked_selected_profile_confirmation as confirm
import run_hybrid_random_control as random_control
import run_least_squares_control as ls_control
import analyze_branch_consistency as branch
from fixed_profile_robustness_common import (
    RHO_VALUES,
    prepare_dataset,
    make_base_layer,
    layer_at_rho,
    selected_success_rate,
)

PLAN = "all_path_rho_selector_sweep_v1"
DEFAULT_OUT = ROOT / "outputs" / PLAN
METHODS = ("zero", "selected_mrbi", "hybrid_ls", "hybrid_random", "multistart5")
PAIR_NAMES = (
    ("selected_mrbi", "zero"),
    ("hybrid_ls", "zero"),
    ("hybrid_random", "zero"),
    ("multistart5", "zero"),
    ("selected_mrbi", "hybrid_ls"),
)
TOL = 1e-6


def mean_bool(xs):
    return float(np.mean(np.asarray(xs, dtype=float)))


def solve_case(dataset, seed, dataset_index, rho, lock):
    cfg, Xtr_p, Xte_p, ytr, yte = prepare_dataset(dataset, seed, lock)
    row, cfg_sel = confirm.selected_cfg(
        dataset, replace(cfg, spectral_radius=float(rho)), lock
    )

    base_layer, base_seed = make_base_layer(Xtr_p, cfg)
    layer, base_rho, actual_rho = layer_at_rho(base_layer, float(rho))

    # Zero
    Z_zero, st_zero = bench.solve_zero_features(Xte_p, layer, cfg)
    Z_zero = np.asarray(Z_zero, dtype=np.float64)

    # Selected MRBI-family path, fixed from development
    Z_mrbi, st_mrbi = branch.solve_selected_test(
        Xte_p, layer, cfg_sel, row, seed
    )
    Z_mrbi = np.asarray(Z_mrbi, dtype=np.float64)

    # Matched random candidate
    Z_random, st_random = random_control.solve_random_policy(
        Xte_p,
        layer,
        cfg_sel,
        row,
        seed + 9000 + 100 * dataset_index,
    )
    Z_random = np.asarray(Z_random, dtype=np.float64)

    # Matched bounded least-squares candidate
    Z_ls, st_ls = ls_control.solve_ls_policy(Xte_p, layer, cfg_sel, row)
    Z_ls = np.asarray(Z_ls, dtype=np.float64)

    # Multistart + strict-root multiplicity
    n_starts = int(lock["confirmation_plan"]["random_multistart"]["n_random_starts"])
    Z_ms, ms_details = branch.solve_multistart_detailed(
        Xte_p,
        layer,
        cfg,
        seed + 7000 + 100 * dataset_index,
        n_starts=n_starts,
        radius=1.0,
    )
    Z_ms = np.asarray(Z_ms, dtype=np.float64)

    reps = {
        "zero": Z_zero,
        "selected_mrbi": Z_mrbi,
        "hybrid_ls": Z_ls,
        "hybrid_random": Z_random,
        "multistart5": Z_ms,
    }

    success = {
        "zero": float(st_zero["zero_success_rate"]),
        "selected_mrbi": float(selected_success_rate(row["kind"], st_mrbi)),
        "hybrid_ls": float(st_ls["success_rate"]),
        "hybrid_random": float(st_random["success_rate"]),
        "multistart5": float(np.mean([d["selected_success"] for d in ms_details])),
    }

    pairs = {}
    pair_arrays = {}
    for a, b in PAIR_NAMES:
        vals = np.asarray(
            [branch.rel_distance(x, y) for x, y in zip(reps[a], reps[b])],
            dtype=float,
        )
        pair_arrays[f"{a}_vs_{b}"] = vals
        pairs[f"{a}_vs_{b}"] = {
            "agreement_1e6": float(np.mean(vals <= TOL)),
            "mean_rel_distance": float(np.mean(vals)),
            "median_rel_distance": float(np.median(vals)),
        }

    distinct = np.asarray(
        [d["distinct_strict_roots_1e6"] for d in ms_details], dtype=int
    )
    strict_count = np.asarray(
        [d["strict_success_count"] for d in ms_details], dtype=int
    )

    # Input-drive strength for later exploratory analysis.
    drive = np.asarray(
        [np.linalg.norm(layer.U @ x + layer.b) for x in np.asarray(Xte_p)],
        dtype=float,
    )

    sample_rows = []
    for i in range(len(Xte_p)):
        sr = {
            "dataset": dataset,
            "seed": int(seed),
            "rho": float(rho),
            "position": int(i),
            "label": int(yte[i]),
            "input_drive_norm": float(drive[i]),
            "multistart_strict_success_count": int(strict_count[i]),
            "multistart_distinct_strict_roots_1e6": int(distinct[i]),
            "multistart_multiple_strict_roots_1e6": int(distinct[i] >= 2),
            "multistart_selected_same_zero_1e6": int(
                ms_details[i]["selected_same_as_zero_1e6"]
            ),
            "zero_success_from_multistart_probe": int(ms_details[i]["zero_success"]),
            "multistart_selected_success": int(ms_details[i]["selected_success"]),
        }
        for key, vals in pair_arrays.items():
            sr[f"rel_distance_{key}"] = float(vals[i])
        sample_rows.append(sr)

    return {
        "dataset": dataset,
        "seed": int(seed),
        "rho": float(rho),
        "base_layer_seed": int(base_seed),
        "base_rho_actual": float(base_rho),
        "rho_actual": float(actual_rho),
        "selected_method": row["method"],
        "selected_kind": row["kind"],
        "selected_profile": row["profile"],
        "selected_hybrid": row["hybrid"],
        "n_test": int(len(Xte_p)),
        "success": success,
        "pairs": pairs,
        "multiplicity": {
            "mean_distinct_strict_roots_1e6": float(np.mean(distinct)),
            "fraction_multiple_strict_roots_1e6": float(np.mean(distinct >= 2)),
            "mean_strict_successful_starts": float(np.mean(strict_count)),
        },
        "sample_rows": sample_rows,
    }


def aggregate(cases):
    rhos = sorted({float(r["rho"]) for r in cases})
    out = {"rhos": rhos, "by_rho": {}}
    for rho in rhos:
        rr = [r for r in cases if float(r["rho"]) == rho]
        if not rr:
            continue
        entry = {
            "n_dataset_seed_runs": len(rr),
            "success": {},
            "pairs": {},
            "multiplicity": {},
        }
        for method in METHODS:
            entry["success"][method] = float(
                np.mean([r["success"][method] for r in rr])
            )
        for pair in rr[0]["pairs"]:
            entry["pairs"][pair] = {}
            for key in rr[0]["pairs"][pair]:
                entry["pairs"][pair][key] = float(
                    np.mean([r["pairs"][pair][key] for r in rr])
                )
        for key in rr[0]["multiplicity"]:
            entry["multiplicity"][key] = float(
                np.mean([r["multiplicity"][key] for r in rr])
            )
        out["by_rho"][f"{rho:.2f}"] = entry
    return out


def write_outputs(out_dir, cases):
    summary = {
        "plan": PLAN,
        "analysis_role": "post-confirmation numerical selector diagnostic",
        "methods": list(METHODS),
        "state_match_threshold": TOL,
        "aggregate": aggregate(cases),
        "cases": [
            {k: v for k, v in r.items() if k != "sample_rows"} for r in cases
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    sample_rows = []
    for r in cases:
        sample_rows.extend(r["sample_rows"])
    if sample_rows:
        with (out_dir / "sample_rows.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
            w.writeheader()
            w.writerows(sample_rows)


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
            raise SystemExit("seed must be one of confirmation seeds 40-49")
        seeds = [args.seed]
    if args.rho is not None:
        if float(args.rho) not in rhos:
            raise SystemExit(f"rho must be one of {rhos}")
        rhos = [float(args.rho)]

    if args.dry_run:
        print(
            "ALL_PATH_RHO_SELECTOR_DRY_RUN_OK "
            f"datasets={len(datasets)} seeds={seeds} rhos={rhos} "
            "qnn=false profiles_fixed=true target_rho_in_seed=false",
            flush=True,
        )
        return

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "checkpoint.json"

    cases = []
    if checkpoint.exists():
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        cases = payload.get("cases", [])

    completed = {
        (r["dataset"], int(r["seed"]), float(r["rho"])) for r in cases
    }

    done_now = 0
    for rho in rhos:
        for seed in seeds:
            for dataset_index, dataset in enumerate(datasets):
                key = (dataset, int(seed), float(rho))
                if key in completed:
                    print(
                        f"SKIP dataset={dataset} seed={seed} rho={rho:.2f}",
                        flush=True,
                    )
                    continue
                print(
                    f"RHO_SELECTOR_START dataset={dataset} seed={seed} rho={rho:.2f}",
                    flush=True,
                )
                result = solve_case(
                    dataset, seed, dataset_index, rho, lock
                )
                cases.append(result)
                completed.add(key)
                checkpoint.write_text(
                    json.dumps(
                        {"plan": PLAN, "cases": cases},
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(
                    "RHO_SELECTOR_DONE "
                    f"dataset={dataset} seed={seed} rho={rho:.2f} "
                    f"succ_zero={result['success']['zero']:.3f} "
                    f"succ_mrbi={result['success']['selected_mrbi']:.3f} "
                    f"succ_ls={result['success']['hybrid_ls']:.3f} "
                    f"succ_random={result['success']['hybrid_random']:.3f} "
                    f"succ_ms={result['success']['multistart5']:.3f} "
                    f"multi={result['multiplicity']['fraction_multiple_strict_roots_1e6']:.3f}",
                    flush=True,
                )
                done_now += 1
                if args.max_cases is not None and done_now >= args.max_cases:
                    write_outputs(out_dir, cases)
                    print("STOPPED_AT_MAX_CASES", flush=True)
                    return

    write_outputs(out_dir, cases)
    print(f"ALL_PATH_RHO_SELECTOR_SWEEP_OK {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
