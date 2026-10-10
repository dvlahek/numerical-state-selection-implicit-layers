"""Verify tracked solver-selector and downstream diagnostic summaries."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"
TOL = 1e-12


def load(name: str):
    return json.loads((FINAL / name).read_text(encoding="utf-8"))


def close(label: str, actual: float, expected: float, tol: float = TOL) -> None:
    if not np.isclose(float(actual), float(expected), atol=tol, rtol=0.0):
        raise AssertionError(f"{label}: {actual} != {expected}")


def main() -> None:
    rho = load("all_path_rho_selector_summary.json")
    if rho["plan"] != "all_path_rho_selector_sweep_v1":
        raise AssertionError("Unexpected all-path rho selector plan")
    if tuple(rho["aggregate"]["rhos"]) != (1.2, 1.6, 2.0, 2.25):
        raise AssertionError("Unexpected rho grid")
    by = rho["aggregate"]["by_rho"]
    close("rho1.2 multiplicity", by["1.20"]["multiplicity"]["fraction_multiple_strict_roots_1e6"], 0.012026245359578693)
    close("rho2 multiplicity", by["2.00"]["multiplicity"]["fraction_multiple_strict_roots_1e6"], 0.27221736596736595)
    close("rho2.25 multiplicity", by["2.25"]["multiplicity"]["fraction_multiple_strict_roots_1e6"], 0.38774173357506686)
    close("rho2 Zero strict success", by["2.00"]["success"]["zero"], 0.7160731244064577)
    close("rho2 Multistart strict success", by["2.00"]["success"]["multistart5"], 0.9924701070534405)
    close("rho2.25 Multistart-Zero agreement", by["2.25"]["pairs"]["multistart5_vs_zero"]["agreement_1e6"], 0.5222017180350513)

    flip = load("rho_multistart_prediction_flip_summary.json")
    if flip["plan"] != "rho_multistart_prediction_flip_v1":
        raise AssertionError("Unexpected rho prediction-flip plan")
    fb = flip["aggregate"]["by_rho"]
    close("rho1.2 state disagreement", fb["1.20"]["mean_dataset_state_change_rate"], 0.028742553742553745)
    close("rho2.25 state disagreement", fb["2.25"]["mean_dataset_state_change_rate"], 0.4777982819649486)
    close("rho1.2 LR disagreement", fb["1.20"]["mean_dataset_prediction_flip_rate"], 0.0020358499525166187)
    close("rho2.25 LR disagreement", fb["2.25"]["mean_dataset_prediction_flip_rate"], 0.030796749546749547)
    close("rho2 Zero LR BA", fb["2.00"]["mean_dataset_zero_ba"], 0.9623699294532627)
    close("rho2 Multistart LR BA", fb["2.00"]["mean_dataset_multistart5_ba"], 0.9623743386243385)

    classical = load("all_path_classical_readouts_summary.json")
    if classical["plan"] != "all_path_classical_readouts_rho2_v1":
        raise AssertionError("Unexpected all-path classical readout plan")
    close("logreg Zero", classical["readouts"]["logreg"]["methods"]["zero"]["dataset_mean_ba"], 0.9623699294532627)
    close("logreg Hybrid-LS", classical["readouts"]["logreg"]["methods"]["hybrid_ls"]["dataset_mean_ba"], 0.9649889770723102)
    close("svm Hybrid-LS", classical["readouts"]["svm_rbf"]["methods"]["hybrid_ls"]["dataset_mean_ba"], 0.9542438271604939)

    rescue = load("sample_level_rescue_summary.json")
    if rescue["plan"] != "sample_level_rescue_analysis_v1":
        raise AssertionError("Unexpected sample-level rescue plan")
    close("GB rescue effect", rescue["readouts"]["gb"]["dataset_mean_rescue_delta_multistart_minus_zero"], 0.024040616859034428)
    signs = rescue["readouts"]["gb"]["rescue_positive_neutral_negative"]
    if signs != {"positive": 7, "neutral": 1, "negative": 1}:
        raise AssertionError(f"Unexpected GB rescue signs: {signs}")

    qnn = load("qnn_multiseed_all_paths_summary.json")
    if qnn["plan"] != "qnn_multiseed_all_paths_v1":
        raise AssertionError("Unexpected QNN all-path plan")
    if qnn["replicates"] != 4 or qnn["n_qnn_trainings"] != 1800:
        raise AssertionError("Unexpected QNN multi-seed dimensions")
    if not qnn["canonical_reproduction_passed"]:
        raise AssertionError("Canonical all-path QNN reproduction failed")
    close("four-seed Zero BA", qnn["method_summary"]["zero"]["mean_over_qnn_seeds"], 0.9298853615520284)
    close("four-seed MRBI BA", qnn["method_summary"]["selected_mrbi"]["mean_over_qnn_seeds"], 0.9321274250440919)
    close("four-seed MRBI-Zero delta", qnn["pair_summary"]["selected_mrbi_minus_zero"]["ensemble_mean_delta"], 0.002242063492063462)
    close("four-seed MRBI-LS delta", qnn["pair_summary"]["selected_mrbi_minus_hybrid_ls"]["ensemble_mean_delta"], 0.0015531305114638602)

    sq = load("selection_quality_qnn_both_strict_summary.json")
    if sq["plan"] != "selection_quality_qnn_both_strict_v1" or sq["qnn_replicates"] != 4:
        raise AssertionError("Unexpected strict-root QNN selection-quality plan")
    close("strict QNN MRBI delta", sq["results"]["selected_mrbi"]["dataset_equal_mean_qnn_accuracy_delta"], 0.03841177744947586)
    close("strict QNN LS delta", sq["results"]["hybrid_ls"]["dataset_equal_mean_qnn_accuracy_delta"], 0.029382779643196306)
    close("strict QNN Random delta", sq["results"]["hybrid_random"]["dataset_equal_mean_qnn_accuracy_delta"], -0.07481386283469617)
    close("strict QNN Multistart delta", sq["results"]["multistart5"]["dataset_equal_mean_qnn_accuracy_delta"], -0.07525869832198613)
    if sq["results"]["multistart5"]["positive_neutral_negative_datasets"] != {"positive": 0, "neutral": 0, "negative": 9}:
        raise AssertionError("Unexpected Multistart strict-root QNN signs")

    geom = load("local_feature_geometry_summary.json")
    if geom["plan"] != "local_feature_regularity_robustness_v1":
        raise AssertionError("Unexpected local geometry plan")
    if geom["reported_view"] != "raw strict-only neighbor pairs" or geom["representative_k"] != 3:
        raise AssertionError("Unexpected local geometry view")
    g = geom["k3_dataset_equal"]
    close("strict geometry MRBI p90", g["selected_mrbi"]["p90_amplification"], 4.920971987393762)
    close("strict geometry MRBI p95", g["selected_mrbi"]["p95_amplification"], 5.933256750329818)
    close("strict geometry LS p90", g["hybrid_ls"]["p90_amplification"], 4.819808724956928)
    close("strict geometry Random p90", g["hybrid_random"]["p90_amplification"], 5.376572118055266)
    close("strict geometry Multistart p90", g["multistart5"]["p90_amplification"], 5.849753646765083)
    if geom["raw_strict_only_ordering"]["3"]["p90"] != [
        "hybrid_ls", "selected_mrbi", "hybrid_random", "multistart5"
    ]:
        raise AssertionError("Unexpected strict-only p90 ordering")

    print("Selector diagnostic checks passed.")


if __name__ == "__main__":
    main()
