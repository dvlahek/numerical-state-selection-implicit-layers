"""Verify tracked post-confirmation diagnostic summaries."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"
BRANCH = FINAL / "branch_state_summary.json"
QNN_SEED = FINAL / "qnn_seed_sensitivity_summary.json"
CONF_JSON = FINAL / "confirmation_summary.json"
CONF_CSV = FINAL / "confirmation_dataset_summary.csv"

DATASETS = (
    "breast_cancer",
    "wine_binary",
    "wine_0_vs_2",
    "wine_1_vs_2",
    "digits_1_vs_7",
    "digits_2_vs_7",
    "digits_3_vs_8",
    "digits_4_vs_9",
    "digits_5_vs_6",
)
TOL = 1e-12


def close(label: str, actual: float, expected: float, tol: float = TOL) -> None:
    if not np.isclose(float(actual), float(expected), atol=tol, rtol=0.0):
        raise AssertionError(f"{label}: {actual} != {expected}")


def main() -> None:
    branch = json.loads(BRANCH.read_text(encoding="utf-8"))
    qnn = json.loads(QNN_SEED.read_text(encoding="utf-8"))
    confirmation = json.loads(CONF_JSON.read_text(encoding="utf-8"))
    conf = pd.read_csv(CONF_CSV)

    if branch["plan"] != "branch_consistency_analysis_v1":
        raise AssertionError("Unexpected returned-state diagnostic plan")
    if tuple(branch["confirmation_seeds"]) != tuple(range(40, 50)):
        raise AssertionError("Returned-state seed set changed")
    if branch["n_dataset_seed_runs"] != 90:
        raise AssertionError("Returned-state run count changed")
    if tuple(branch["same_state_thresholds"]) != (1e-6, 1e-4):
        raise AssertionError("Returned-state thresholds changed")

    pairs = branch["pairwise_state_summary"]
    close("MRBI-Zero state agreement", pairs["mrbi_vs_zero"]["same_rate_1e-06"], 0.7381555944055945)
    close("LS-Zero state agreement", pairs["hybrid_ls_vs_zero"]["same_rate_1e-06"], 0.7565640378140379)
    close("Random-Zero state agreement", pairs["hybrid_random_vs_zero"]["same_rate_1e-06"], 0.6966837175170509)
    close("Multistart-Zero state agreement", pairs["multistart5_vs_zero"]["same_rate_1e-06"], 0.5897209272209273)
    close("MRBI-LS state agreement", pairs["mrbi_vs_hybrid_ls"]["same_rate_1e-06"], 0.7523558231891563)
    close("MRBI-LS mean distance", pairs["mrbi_vs_hybrid_ls"]["mean_rel_distance"], 0.023120417694982607)

    roots = branch["multistart_root_summary"]
    close("multiple strict-root fraction", roots["fraction_multiple_strict_roots_1e6"], 0.27221736596736595)
    close("mean distinct strict roots", roots["mean_distinct_strict_roots_1e6"], 1.4364300051800052)

    subset = branch["multiple_root_subset"]
    if subset["n_test_samples_total"] != 3960:
        raise AssertionError("Pooled returned-state sample count changed")
    if subset["n_samples_with_at_least_two_strict_roots_1e6"] != 1080:
        raise AssertionError("Multiple-root subset size changed")
    close("multiple-root MRBI-LS agreement", subset["same_state_rate_mrbi_vs_hybrid_ls_1e6"], 0.8462962962962963)
    close("multiple-root multistart-Zero agreement", subset["same_state_rate_multistart5_vs_zero_1e6"], 0.4462962962962963)

    if qnn["plan"] != "qnn_zero_feature_noise_floor_v1":
        raise AssertionError("Unexpected QNN readout-seed diagnostic plan")
    if qnn["replicates"] != 4:
        raise AssertionError("QNN readout-seed replicate count changed")

    noise = qnn["confirmation_noise_floor"]
    rows = noise["dataset_rows"]
    if tuple(row["dataset"] for row in rows) != DATASETS:
        raise AssertionError("QNN readout-seed dataset set/order changed")

    sds = np.asarray([row["sd_across_qnn_seed_means"] for row in rows], dtype=float)
    ranges = np.asarray([row["range_across_qnn_seed_means"] for row in rows], dtype=float)
    canonical = np.asarray([row["canonical_dataset_mean"] for row in rows], dtype=float)

    close("mean dataset-level QNN SD", sds.mean(), noise["mean_dataset_level_sd"])
    close("median dataset-level QNN SD", np.median(sds), noise["median_dataset_level_sd"])
    close("mean dataset-level QNN range", ranges.mean(), noise["mean_dataset_level_range"])

    if tuple(conf["dataset"]) != DATASETS:
        raise AssertionError("Confirmation dataset order changed")
    if not np.allclose(canonical, conf["zero"].to_numpy(float), atol=TOL, rtol=0.0):
        raise AssertionError("Canonical Zero-QNN means do not match confirmation results")

    dev = qnn["development_zero"]
    dev_rows = dev["dataset_summary"]
    if tuple(row["dataset"] for row in dev_rows) != DATASETS:
        raise AssertionError("Development Zero dataset set/order changed")
    dev_means = np.asarray([row["zero_dev_ba_mean"] for row in dev_rows], dtype=float)
    close("development Zero overall mean", dev_means.mean(), dev["overall_dataset_mean"])
    close(
        "reported development selected mean",
        confirmation["overall"]["development_selected_ba_mean"],
        0.9577072311111112,
    )

    print("Post-confirmation diagnostic checks passed.")


if __name__ == "__main__":
    main()
