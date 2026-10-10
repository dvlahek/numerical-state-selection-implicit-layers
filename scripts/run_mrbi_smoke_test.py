from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import mrbi


def main():
    rng = np.random.default_rng(0)
    d = 6
    dx = 3
    W = rng.normal(size=(d, d))
    rho = max(abs(np.linalg.eigvals(W)))
    W = W / rho * 1.2
    U = rng.normal(scale=0.5, size=(d, dx))
    b = rng.normal(scale=0.05, size=d)
    x = rng.normal(size=dx)

    layer = mrbi.ImplicitTanhLayer(W=W, U=U, b=b)
    res = mrbi.hybrid_mrbi_solve_tanh_layer(layer, x, rng=rng)

    print({
        "success": bool(res.success),
        "residual": float(res.residual),
        "used_mrbi": bool(res.used_mrbi),
        "trigger_reason": res.trigger_reason,
    })

    if not np.isfinite(res.residual):
        raise SystemExit("Non-finite residual in smoke test.")


if __name__ == "__main__":
    main()
