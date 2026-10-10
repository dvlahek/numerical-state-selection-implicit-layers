"""Run fixed classical readouts on all five cached rho=2 feature paths."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_locked_selected_profile_confirmation as confirm
import run_qnn_multiseed_all_paths as allpaths

PLAN = "all_path_classical_readouts_rho2_v1"
OUT = ROOT / "outputs" / PLAN
FEATURE_ROOT = (
    ROOT / "outputs" / "qnn_multiseed_all_paths_v1" / "features"
)
METHODS = (
    "zero", "selected_mrbi", "hybrid_ls", "hybrid_random", "multistart5"
)
READOUTS = ("logreg", "svm_rbf", "mlp", "rf", "gb")


def make_model(name, seed):
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=3000),
        )
    if name == "svm_rbf":
        return make_pipeline(
            StandardScaler(),
            SVC(C=1.0, gamma="scale", probability=True),
        )
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                learning_rate_init=1e-3,
                alpha=1e-4,
                max_iter=3000,
                random_state=seed,
            ),
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=seed,
        )
    if name == "gb":
        return GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=seed,
        )
    raise ValueError(name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = confirm.load_lock()
    datasets = list(lock["development_evidence"]["datasets"])
    seeds = list(lock["confirmation_plan"]["seeds"])

    rows = []
    for seed in seeds:
        for dataset in datasets:
            path = FEATURE_ROOT / f"seed{seed}" / f"{dataset}.npz"
            if not path.exists():
                raise SystemExit(f"Missing cached feature file: {path}")
            ytr, yte, reps = allpaths.load_feature_cache(path)
            for readout in READOUTS:
                for method in METHODS:
                    Xtr, Xte = reps[method]
                    model = make_model(readout, seed)
                    model.fit(Xtr, ytr)
                    pred = model.predict(Xte)
                    ba = float(balanced_accuracy_score(yte, pred))
                    rows.append({
                        "dataset": dataset,
                        "seed": int(seed),
                        "readout": readout,
                        "method": method,
                        "balanced_accuracy": ba,
                    })
                    print(
                        f"CLASSICAL_DONE dataset={dataset} seed={seed} "
                        f"readout={readout} method={method} ba={ba:.4f}",
                        flush=True,
                    )

    with (OUT / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {"plan": PLAN, "readouts": {}}
    for readout in READOUTS:
        summary["readouts"][readout] = {"methods": {}, "datasets": {}}
        for method in METHODS:
            per_ds = []
            for dataset in datasets:
                vals = [
                    r["balanced_accuracy"] for r in rows
                    if r["readout"] == readout
                    and r["method"] == method
                    and r["dataset"] == dataset
                ]
                if len(vals) != 10:
                    raise RuntimeError(
                        f"Expected 10 rows for {readout}/{method}/{dataset}"
                    )
                m = float(np.mean(vals))
                per_ds.append(m)
                summary["readouts"][readout]["datasets"].setdefault(dataset, {})[
                    method
                ] = m
            summary["readouts"][readout]["methods"][method] = {
                "dataset_mean_ba": float(np.mean(per_ds))
            }

        pairs = [
            ("selected_mrbi", "zero"),
            ("hybrid_ls", "zero"),
            ("hybrid_random", "zero"),
            ("multistart5", "zero"),
            ("selected_mrbi", "hybrid_ls"),
        ]
        ps = {}
        for a, b in pairs:
            deltas = []
            for dataset in datasets:
                d = summary["readouts"][readout]["datasets"][dataset]
                deltas.append(float(d[a] - d[b]))
            arr = np.asarray(deltas)
            ps[f"{a}_minus_{b}"] = {
                "mean_delta": float(np.mean(arr)),
                "median_delta": float(np.median(arr)),
                "positive": int(np.sum(arr > 1e-12)),
                "neutral": int(np.sum(np.abs(arr) <= 1e-12)),
                "negative": int(np.sum(arr < -1e-12)),
                "dataset_deltas": deltas,
            }
        summary["readouts"][readout]["pairs"] = ps

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"ALL_PATH_CLASSICAL_READOUTS_OK {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
