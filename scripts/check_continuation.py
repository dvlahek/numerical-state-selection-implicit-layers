"""Verify the MRBI continuation schedule without running a QNN benchmark."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import mrbi


def check_schedule(config, expected_scales, expected_iterations):
    def F(z, x):
        return np.asarray(z - x, dtype=float)

    def J(z, x):
        return np.eye(len(z))

    optimizer = mrbi.MRBIOptimizer(
        F=F, J=J, x=np.array([0.1, -0.2]), dim=2,
        config=config, rng=np.random.default_rng(11),
    )
    recorded = []

    def record_stage(z0, sigma, maxiter, *, detector_override=None):
        recorded.append((float(sigma), int(maxiter), detector_override))
        return np.asarray(z0, dtype=float).copy()

    optimizer._optimize_single_scale = record_stage
    candidate = optimizer.optimize()
    actual_scales = [v[0] for v in recorded]
    actual_iterations = [v[1] for v in recorded]
    if actual_scales != expected_scales:
        raise AssertionError(f"Continuation scales: {actual_scales} != {expected_scales}")
    if actual_iterations != expected_iterations:
        raise AssertionError(
            f"Stage iteration budgets: {actual_iterations} != {expected_iterations}"
        )
    if not all(v[2] is True for v in recorded):
        raise AssertionError("Detector override changed unexpectedly")
    if candidate.last_sigma != expected_scales[-1]:
        raise AssertionError("Incorrect final sigma")


def main():
    cfg = mrbi.MRBIConfig(
        sigmas=(0.5, 0.1, 0.02),
        gamma=0.0,
        use_detector=False,
        mc_samples=0,
        maxiter_per_scale=7,
        refinement_iters=3,
        use_refinement=True,
    )
    check_schedule(cfg, [0.5, 0.1, 0.02, 0.02], [7, 7, 7, 3])
    cfg.use_refinement = False
    check_schedule(cfg, [0.5, 0.1, 0.02], [7, 7, 7])
    cfg.use_continuation = False
    cfg.use_refinement = True
    check_schedule(cfg, [0.02, 0.02], [7, 3])
    cfg.use_continuation = True

    for invalid in [(0.1, 0.5), (0.5, 0.5), (0.5, 0.0), (), (float("nan"),)]:
        cfg.sigmas = invalid
        try:
            check_schedule(cfg, [], [])
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid sigma schedule accepted: {invalid}")

    print("MRBI continuation tests passed: descending stages, refinement, validation.")


if __name__ == "__main__":
    main()
