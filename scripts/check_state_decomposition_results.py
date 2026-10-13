"""Verify the tracked common-mask state-retention aggregate summary."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "final" / "common_mask_state_decomposition_summary.csv"
TOL = 5e-5

EXPECTED = {
    "selected_mrbi": {
        "n_same": 491,
        "n_different": 183,
        "lr_same": -0.0012,
        "lr_different": 0.0149,
        "svm_same": 0.0236,
        "svm_different": 0.0622,
        "mlp_same": -0.0035,
        "mlp_different": 0.0639,
        "rf_same": 0.0162,
        "rf_different": -0.0034,
        "gb_same": 0.0025,
        "gb_different": -0.0071,
        "qnn_same": -0.0088,
        "qnn_different": 0.0384,
    },
    "hybrid_ls": {
        "n_same": 482,
        "n_different": 192,
        "lr_same": -0.0012,
        "lr_different": 0.0207,
        "svm_same": 0.0245,
        "svm_different": 0.0469,
        "mlp_same": -0.0031,
        "mlp_different": 0.0507,
        "rf_same": 0.0113,
        "rf_different": 0.0104,
        "gb_same": -0.0037,
        "gb_different": 0.0148,
        "qnn_same": -0.0151,
        "qnn_different": 0.0294,
    },
    "hybrid_random": {
        "n_same": 461,
        "n_different": 213,
        "lr_same": -0.0062,
        "lr_different": -0.0188,
        "svm_same": 0.0077,
        "svm_different": -0.0431,
        "mlp_same": -0.0011,
        "mlp_different": -0.0507,
        "rf_same": 0.0068,
        "rf_different": -0.0749,
        "gb_same": -0.0058,
        "gb_different": -0.0938,
        "qnn_same": -0.0165,
        "qnn_different": -0.0748,
    },
    "multistart5": {
        "n_same": 163,
        "n_different": 511,
        "lr_same": 0.0034,
        "lr_different": -0.0198,
        "svm_same": -0.0009,
        "svm_different": -0.0035,
        "mlp_same": -0.0078,
        "mlp_different": -0.0217,
        "rf_same": 0.0054,
        "rf_different": -0.0440,
        "gb_same": -0.0479,
        "gb_different": -0.0431,
        "qnn_same": 0.0015,
        "qnn_different": -0.0753,
    },
}


def main() -> None:
    df = pd.read_csv(RESULT)
    expected_order = tuple(EXPECTED)
    if tuple(df["selector"]) != expected_order:
        raise AssertionError("Unexpected selector set/order in state decomposition summary")

    for selector, expected in EXPECTED.items():
        row = df[df["selector"] == selector].iloc[0]
        if int(row["n_same"]) != expected["n_same"]:
            raise AssertionError(f"{selector} n_same changed")
        if int(row["n_different"]) != expected["n_different"]:
            raise AssertionError(f"{selector} n_different changed")
        if int(row["n_same"]) + int(row["n_different"]) != 674:
            raise AssertionError(f"{selector} common-mask count is not 674")

        for column, value in expected.items():
            if column.startswith("n_"):
                continue
            if not np.isclose(float(row[column]), float(value), atol=TOL, rtol=0.0):
                raise AssertionError(
                    f"{selector} {column}: {row[column]} != {value}"
                )

    print("State-retention aggregate checks passed.")


if __name__ == "__main__":
    main()
