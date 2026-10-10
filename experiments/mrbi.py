"""
mrbi.py

Implementation of Multiscale Residual-Based Initialization (MRBI) and the
hybrid wrapper used for nonlinear equilibrium solving.

The module contains the numerical method only. Dataset loading, readout
training, and benchmark orchestration are handled elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, List, Dict
import time

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, root

Array = NDArray[np.float64]
ResidualFn = Callable[[Array, Array], Array]
JacobianFn = Callable[[Array, Array], Array]


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def _as_float_array(x: Array | Sequence[float]) -> Array:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Expected a 1D array.")
    return arr


def _norm(v: Array) -> float:
    if not np.all(np.isfinite(v)):
        return float("inf")
    return float(np.linalg.norm(v))


def _safe_norm(v: Array, large: float = 1e12) -> float:
    n = _norm(v)
    return n if np.isfinite(n) else large


def _default_rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


# ---------------------------------------------------------------------
# Tanh implicit layer used in the benchmark
# ---------------------------------------------------------------------

@dataclass
class ImplicitTanhLayer:
    """
    Wrapper for the equilibrium model

        z* = tanh(W z* + U x + b)

    with residual and Jacobian construction.
    """
    W: Array
    U: Array
    b: Array

    def __post_init__(self) -> None:
        self.W = np.asarray(self.W, dtype=np.float64)
        self.U = np.asarray(self.U, dtype=np.float64)
        self.b = np.asarray(self.b, dtype=np.float64)

        if self.W.ndim != 2 or self.W.shape[0] != self.W.shape[1]:
            raise ValueError("W must be square with shape (d, d).")

        d = self.W.shape[0]
        if self.U.ndim != 2 or self.U.shape[0] != d:
            raise ValueError("U must have shape (d, d_x).")
        if self.b.shape != (d,):
            raise ValueError("b must have shape (d,).")

    @property
    def d(self) -> int:
        return self.W.shape[0]

    def residual(self, z: Array, x: Array) -> Array:
        z = _as_float_array(z)
        x = _as_float_array(x)
        a = self.W @ z + self.U @ x + self.b
        return z - np.tanh(a)

    def jacobian(self, z: Array, x: Array) -> Array:
        z = _as_float_array(z)
        x = _as_float_array(x)
        a = self.W @ z + self.U @ x + self.b
        g = 1.0 - np.tanh(a) ** 2
        return np.eye(self.d, dtype=np.float64) - g[:, None] * self.W


def make_residual_and_jacobian(layer: ImplicitTanhLayer) -> Tuple[ResidualFn, JacobianFn]:
    return layer.residual, layer.jacobian


# ---------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------

@dataclass
class RootSolveConfig:
    """
    Configuration for the downstream nonlinear root solver.
    """
    method: str = "hybr"
    tol: float = 1e-10
    success_residual_tol: float = 1e-8


@dataclass
class MRBIConfig:
    """
    Configuration for MRBI candidate construction.
    """
    sigmas: Tuple[float, ...] = (0.5, 0.1, 0.03, 0.01)

    alpha: float = 1.0
    beta: float = 0.01
    gamma: float = 0.01
    lambda_zero: float = 0.2
    lambda_relative: float = 1.0
    epsilon: float = 1e-8

    search_radius: float = 0.5
    mc_samples: int = 16
    maxiter_per_scale: int = 120
    refinement_iters: int = 120

    use_continuation: bool = True
    # Ablation only: keep the number and budgets of the stages, but
    # optimize at the final sigma on every stage. The default is unchanged.
    repeat_final_sigma: bool = False
    use_refinement: bool = True
    use_detector: bool = True
    use_newton_term: bool = True

    n_starts: int = 1
    random_start_scale: float = 0.05

    # numerical regularization for ill-conditioned Newton proxy
    newton_damping: float = 1e-6
    optimizer_ftol: float = 1e-12

    # variance reduction / determinism for detector estimation
    use_fixed_probes_per_scale: bool = True
    use_antithetic_sampling: bool = True


@dataclass
class HybridConfig:
    """
    Triggering and acceptance rules for the hybrid MRBI fallback.
    """
    tau_r: float = 1e-8                # absolute floor for trigger
    tau_sigma: float = 0.03
    accept_factor_vs_zero: float = 1.05

    # effective trigger threshold: max(tau_r, residual_trigger_scale * ||F(0;x)||)
    use_relative_residual_trigger: bool = True
    residual_trigger_scale: float = 0.1

    # additional zero-solve reliability checks
    trigger_on_solver_failure: bool = True
    trigger_on_bad_jacobian: bool = True

    # tie-breaking when residuals are very close
    tie_residual_rtol: float = 0.02
    prefer_better_conditioning_on_tie: bool = True

    # when zero solve fails badly, allow accepting any finite successful MRBI solve
    accept_any_finite_success_if_zero_failed: bool = True


@dataclass
class RootResult:
    success: bool
    z_star: Array
    residual: float
    nfev: int
    runtime_sec: float
    message: str = ""
    solver_success_flag: bool = False
    has_finite_z: bool = True
    has_finite_residual: bool = True


@dataclass
class MRBICandidate:
    z_init: Array
    objective_value: float
    last_sigma: float
    n_objective_calls: int


@dataclass
class HybridSolveResult:
    success: bool
    z_star: Array
    residual: float
    used_mrbi: bool
    trigger_reason: str
    zero_result: RootResult
    final_result: RootResult
    mrbi_candidate: Optional[MRBICandidate] = None
    sigma_min_zero: Optional[float] = None
    sigma_min_det: Optional[float] = None
    total_runtime_sec: Optional[float] = None


# ---------------------------------------------------------------------
# Root solve and Jacobian conditioning
# ---------------------------------------------------------------------

def solve_root(
    F: ResidualFn,
    J: JacobianFn,
    x: Array,
    z0: Array,
    cfg: RootSolveConfig | None = None,
) -> RootResult:
    """
    Solve F(z; x) = 0 from initialization z0 using scipy.optimize.root.
    Solver exceptions are converted to a failed RootResult.
    """
    cfg = cfg or RootSolveConfig()
    z0 = _as_float_array(z0)
    x = _as_float_array(x)

    t0 = time.perf_counter()

    def fun(z: Array) -> Array:
        z = np.asarray(z, dtype=np.float64)
        out = np.asarray(F(z, x), dtype=np.float64)
        if out.ndim != 1:
            raise ValueError("F must return a 1D residual vector.")
        return out

    def jac(z: Array) -> Array:
        z = np.asarray(z, dtype=np.float64)
        out = np.asarray(J(z, x), dtype=np.float64)
        if out.ndim != 2:
            raise ValueError("J must return a 2D Jacobian matrix.")
        return out

    methods_with_jac = {"hybr", "lm"}

    try:
        if cfg.method in methods_with_jac:
            res = root(fun, z0, jac=jac, method=cfg.method, tol=cfg.tol)
        else:
            res = root(fun, z0, method=cfg.method, tol=cfg.tol)

        z_star = np.asarray(res.x, dtype=np.float64)
        solver_success_flag = bool(getattr(res, "success", False))
        has_finite_z = bool(np.all(np.isfinite(z_star)))

        try:
            F_star = fun(z_star)
            has_finite_residual = bool(np.all(np.isfinite(F_star)))
            residual = _safe_norm(F_star)
        except Exception:
            has_finite_residual = False
            residual = float("inf")

        success = bool(
            solver_success_flag
            and has_finite_z
            and has_finite_residual
            and residual <= cfg.success_residual_tol
        )

        return RootResult(
            success=success,
            z_star=z_star,
            residual=residual,
            nfev=int(getattr(res, "nfev", -1)),
            runtime_sec=time.perf_counter() - t0,
            message=str(getattr(res, "message", "")),
            solver_success_flag=solver_success_flag,
            has_finite_z=has_finite_z,
            has_finite_residual=has_finite_residual,
        )

    except Exception as e:
        z_fail = np.asarray(z0, dtype=np.float64).copy()
        finite_z = bool(np.all(np.isfinite(z_fail)))
        return RootResult(
            success=False,
            z_star=z_fail if finite_z else np.zeros_like(z0),
            residual=float("inf"),
            nfev=-1,
            runtime_sec=time.perf_counter() - t0,
            message=f"{type(e).__name__}: {e}",
            solver_success_flag=False,
            has_finite_z=finite_z,
            has_finite_residual=False,
        )
    
def jacobian_health(J: JacobianFn, x: Array, z: Array) -> tuple[float, bool]:
    """
    Returns:
        sigma_min : smallest singular value if available, else 0.0
        ok        : True when the Jacobian is square and finite
    """
    try:
        Jz = np.asarray(J(z, x), dtype=np.float64)
        if Jz.ndim != 2 or Jz.shape[0] != Jz.shape[1]:
            return 0.0, False
        if not np.all(np.isfinite(Jz)):
            return 0.0, False

        svals = np.linalg.svd(Jz, compute_uv=False)
        if not np.all(np.isfinite(svals)):
            return 0.0, False

        return float(np.min(svals)), True
    except Exception:
        return 0.0, False


def sigma_min_jacobian(J: JacobianFn, x: Array, z: Array) -> float:
    """
    Backward-compatible helper.
    """
    smin, _ = jacobian_health(J, x, z)
    return smin


# ---------------------------------------------------------------------
# MRBI internals
# ---------------------------------------------------------------------

class MRBIOptimizer:
    """
    Builds MRBI candidate initializations for a given residual map F and Jacobian J.
    """

    def __init__(
        self,
        F: ResidualFn,
        J: JacobianFn,
        x: Array,
        dim: int,
        config: MRBIConfig | None = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.F = F
        self.J = J
        self.x = _as_float_array(x)
        self.cfg = config or MRBIConfig()
        self.rng = _default_rng(rng)

        self.d = int(dim)
        if self.d <= 0:
            raise ValueError("dim must be a positive integer.")

        self.objective_calls = 0
        self.zero = np.zeros(self.d, dtype=np.float64)
        self.zero_residual = np.asarray(self.F(self.zero, self.x), dtype=np.float64)
        self.zero_residual_norm = _safe_norm(self.zero_residual)

        # Optional fixed Gaussian probes by sigma to reduce objective noise.
        self._probe_cache: Dict[float, Array] = {}

    def residual(self, z: Array) -> Array:
        return np.asarray(self.F(_as_float_array(z), self.x), dtype=np.float64)

    def jacobian(self, z: Array) -> Array:
        return np.asarray(self.J(_as_float_array(z), self.x), dtype=np.float64)

    def newton_proxy(self, z: Array) -> Array:
        """
        Solve J(z) s = F(z) and return -s.
        Uses Tikhonov-like damping if needed.
        """
        z = _as_float_array(z)
        Fz = self.residual(z)
        Jz = self.jacobian(z)

        if not np.all(np.isfinite(Jz)) or not np.all(np.isfinite(Fz)):
            return np.full_like(z, 1e6, dtype=np.float64)

        try:
            s = np.linalg.solve(Jz, Fz)
            if np.all(np.isfinite(s)):
                return -np.asarray(s, dtype=np.float64)
        except np.linalg.LinAlgError:
            pass

        lam = float(self.cfg.newton_damping)
        I = np.eye(Jz.shape[0], dtype=np.float64)

        try:
            s = np.linalg.solve(Jz + lam * I, Fz)
            if np.all(np.isfinite(s)):
                return -np.asarray(s, dtype=np.float64)
        except np.linalg.LinAlgError:
            pass

        try:
            s, *_ = np.linalg.lstsq(Jz + lam * I, Fz, rcond=None)
            return -np.asarray(s, dtype=np.float64)
        except Exception:
            return np.full_like(z, 1e6, dtype=np.float64)

    def _make_probes(self, sigma: float) -> Array:
        K = int(self.cfg.mc_samples)
        if K <= 0:
            return np.zeros((0, self.d), dtype=np.float64)

        if self.cfg.use_antithetic_sampling and K >= 2:
            half = (K + 1) // 2
            U_half = self.rng.normal(size=(half, self.d)).astype(np.float64)
            U = np.concatenate([U_half, -U_half], axis=0)[:K]
        else:
            U = self.rng.normal(size=(K, self.d)).astype(np.float64)
        return U

    def _get_probes(self, sigma: float) -> Array:
        key = float(sigma)
        if self.cfg.use_fixed_probes_per_scale:
            if key not in self._probe_cache:
                self._probe_cache[key] = self._make_probes(key)
            return self._probe_cache[key]
        return self._make_probes(key)

    def monte_carlo_ratio(self, z: Array, sigma: float) -> float:
        """
        Monte Carlo approximation of the detector term.
        """
        z = _as_float_array(z)
        if not self.cfg.use_detector or self.cfg.gamma == 0.0:
            return 0.0

        U = self._get_probes(float(sigma))
        if U.shape[0] == 0:
            return 0.0

        try:
            Fvals = np.stack([self.residual(z + float(sigma) * u) for u in U], axis=0)
        except Exception:
            return 1e12

        if not np.all(np.isfinite(Fvals)):
            return 1e12

        Phat = np.mean(Fvals, axis=0)
        What = np.mean(Fvals * U, axis=0)

        num = _safe_norm(Phat)
        den = _safe_norm(What) / max(float(sigma), 1e-16) + float(self.cfg.epsilon)
        return float(num / den)

    def objective(
        self,
        z: Array,
        sigma: float,
        *,
        detector_override: Optional[bool] = None,
    ) -> float:
        """
        MRBI objective at a given scale sigma.
        """
        self.objective_calls += 1
        z = _as_float_array(z)

        Fz = self.residual(z)
        Fz_norm = _safe_norm(Fz)

        residual_term = float(self.cfg.alpha) * Fz_norm

        if self.cfg.use_newton_term and self.cfg.beta != 0.0:
            newton_term = float(self.cfg.beta) * _safe_norm(self.newton_proxy(z))
        else:
            newton_term = 0.0

        use_detector_here = self.cfg.use_detector if detector_override is None else bool(detector_override)
        if use_detector_here and self.cfg.gamma != 0.0:
            ratio_term = float(self.cfg.gamma) * self.monte_carlo_ratio(z, sigma)
        else:
            ratio_term = 0.0

        reg_zero = float(self.cfg.lambda_zero) * _safe_norm(z)

        rel_penalty = max(0.0, Fz_norm - self.zero_residual_norm)
        reg_relative = float(self.cfg.lambda_relative) * rel_penalty

        val = residual_term + newton_term + ratio_term + reg_zero + reg_relative
        return float(val if np.isfinite(val) else 1e12)

    def _optimize_single_scale(
        self,
        z0: Array,
        sigma: float,
        maxiter: int,
        *,
        detector_override: Optional[bool] = None,
    ) -> Array:
        """
        Optimize the MRBI objective at a single scale.
        """
        z0 = _as_float_array(z0)
        bounds = [(-self.cfg.search_radius, self.cfg.search_radius) for _ in range(self.d)]

        res = minimize(
            fun=lambda z: self.objective(
                np.asarray(z, dtype=np.float64),
                float(sigma),
                detector_override=detector_override,
            ),
            x0=z0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": float(self.cfg.optimizer_ftol)},
        )

        z_opt = np.asarray(res.x, dtype=np.float64)
        z_opt = np.clip(z_opt, -self.cfg.search_radius, self.cfg.search_radius)
        return z_opt

    def optimize(self) -> MRBICandidate:
        """
        Build an MRBI initialization candidate, optionally with multiple starts.
        """
        scales: Sequence[float] = tuple(float(s) for s in self.cfg.sigmas)
        if not scales or any(not np.isfinite(s) or s <= 0 for s in scales):
            raise ValueError("sigmas must contain positive finite scales.")
        if self.cfg.use_continuation and any(
            earlier <= later for earlier, later in zip(scales, scales[1:])
        ):
            raise ValueError("Continuation sigmas must be strictly decreasing.")
        if int(self.cfg.n_starts) < 1:
            raise ValueError("n_starts must be at least one.")
        if self.cfg.repeat_final_sigma and (
            not self.cfg.use_continuation or not self.cfg.use_fixed_probes_per_scale
        ):
            raise ValueError(
                "The final-sigma ablation requires the full stage schedule "
                "and fixed probes per scale."
            )

        sigmas = scales if self.cfg.use_continuation else (scales[-1],)

        best_z: Optional[Array] = None
        best_val = float("inf")
        best_last_sigma = float(sigmas[-1])

        for start_idx in range(int(self.cfg.n_starts)):
            if start_idx == 0:
                z = np.zeros(self.d, dtype=np.float64)
            else:
                z = self.rng.normal(
                    scale=float(self.cfg.random_start_scale), size=self.d
                ).astype(np.float64)
                z = np.clip(z, -self.cfg.search_radius, self.cfg.search_radius)
            final_sigma = float(sigmas[-1])
            if self.cfg.repeat_final_sigma and self.cfg.use_detector and self.cfg.gamma != 0.0:
                # Consume the same Gaussian draws as continuation: all stages
                # use the same fixed per-sigma probes; only the objective
                # sigma supplied to L-BFGS-B differs. This preserves the RNG
                # position at the start of the next input sample.
                for sigma in sigmas:
                    self._get_probes(float(sigma))
            for sigma in sigmas:
                stage_sigma = final_sigma if self.cfg.repeat_final_sigma else float(sigma)
                z = self._optimize_single_scale(
                    z0=z,
                    sigma=stage_sigma,
                    maxiter=int(self.cfg.maxiter_per_scale),
                    detector_override=True,
                )

            if self.cfg.use_refinement and self.cfg.refinement_iters > 0:
                # Additional optimization pass at the final scale
                final_sigma = float(sigmas[-1])
                z = self._optimize_single_scale(
                    z0=z,
                    sigma=final_sigma,
                    maxiter=int(self.cfg.refinement_iters),
                    detector_override=True,
                )

            val = self.objective(z, float(sigmas[-1]), detector_override=True)
            if val < best_val:
                best_val = val
                best_z = z.copy()
                best_last_sigma = float(sigmas[-1])

        assert best_z is not None, "Internal error: no MRBI candidate produced."

        return MRBICandidate(
            z_init=best_z,
            objective_value=float(best_val),
            last_sigma=best_last_sigma,
            n_objective_calls=int(self.objective_calls),
        )


# ---------------------------------------------------------------------
# Hybrid wrapper
# ---------------------------------------------------------------------

def hybrid_mrbi_solve(
    F: ResidualFn,
    J: JacobianFn,
    x: Array,
    *,
    dim: int,
    mrbi_cfg: MRBIConfig | None = None,
    hybrid_cfg: HybridConfig | None = None,
    root_cfg: RootSolveConfig | None = None,
    rng: Optional[np.random.Generator] = None,
) -> HybridSolveResult:
    """
    Hybrid equilibrium solve:
    1) solve from zero initialization
    2) trigger MRBI if zero solve looks unreliable
    3) solve from MRBI candidate
    4) accept MRBI solution only if it is sufficiently competitive vs zero

    Parameters
    ----------
    F, J:
        Residual map and Jacobian callables with signatures
            F(z, x) -> residual vector
            J(z, x) -> Jacobian matrix
    x:
        Input vector.
    dim:
        Latent dimension. Required for the generic API.
    """
    mrbi_cfg = mrbi_cfg or MRBIConfig()
    hybrid_cfg = hybrid_cfg or HybridConfig()
    root_cfg = root_cfg or RootSolveConfig()
    rng = _default_rng(rng)
    x = _as_float_array(x)

    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be a positive integer.")

    t0 = time.perf_counter()

    z_zero = np.zeros(dim, dtype=np.float64)
    F0 = np.asarray(F(z_zero, x), dtype=np.float64)
    F0_norm = _safe_norm(F0)

    if hybrid_cfg.use_relative_residual_trigger:
        tau_r_eff = max(float(hybrid_cfg.tau_r), float(hybrid_cfg.residual_trigger_scale) * F0_norm)
    else:
        tau_r_eff = float(hybrid_cfg.tau_r)

    zero_result = solve_root(F, J, x, z_zero, cfg=root_cfg)
    s_zero, jacobian_zero_ok = jacobian_health(J, x, zero_result.z_star)

    trigger_reasons: List[str] = []

    if zero_result.residual > tau_r_eff:
        trigger_reasons.append("residual")
    if s_zero < hybrid_cfg.tau_sigma:
        trigger_reasons.append("conditioning")
    if hybrid_cfg.trigger_on_solver_failure and not zero_result.solver_success_flag:
        trigger_reasons.append("solver_failure")
    if hybrid_cfg.trigger_on_bad_jacobian and not jacobian_zero_ok:
        trigger_reasons.append("bad_jacobian")

    if not trigger_reasons:
        total_runtime = time.perf_counter() - t0
        return HybridSolveResult(
            success=zero_result.success,
            z_star=zero_result.z_star,
            residual=zero_result.residual,
            used_mrbi=False,
            trigger_reason="none",
            zero_result=zero_result,
            final_result=zero_result,
            mrbi_candidate=None,
            sigma_min_zero=s_zero,
            sigma_min_det=None,
            total_runtime_sec=total_runtime,
        )

    optimizer = MRBIOptimizer(
        F=F,
        J=J,
        x=x,
        dim=dim,
        config=mrbi_cfg,
        rng=rng,
    )
    candidate = optimizer.optimize()

    det_result = solve_root(F, J, x, candidate.z_init, cfg=root_cfg)
    s_det, jacobian_det_ok = jacobian_health(J, x, det_result.z_star)

    zero_resid = zero_result.residual
    det_resid = det_result.residual

    use_mrbi = False

    # Optional fallback when the zero solve is unreliable.
    if (
        hybrid_cfg.accept_any_finite_success_if_zero_failed
        and not zero_result.solver_success_flag
        and det_result.success
        and np.isfinite(det_resid)
    ):
        use_mrbi = True
    else:
        accept_limit = float(hybrid_cfg.accept_factor_vs_zero) * zero_resid
        competitive = bool(det_resid <= accept_limit)
        clearly_better = bool(
            det_resid < (1.0 - float(hybrid_cfg.tie_residual_rtol)) * zero_resid
        )

        if competitive:
            if clearly_better:
                use_mrbi = True
            else:
                # near tie: use conditioning as a tie-breaker if enabled
                if (
                    hybrid_cfg.prefer_better_conditioning_on_tie
                    and jacobian_det_ok
                    and (s_det > s_zero)
                ):
                    use_mrbi = True
                else:
                    use_mrbi = False

    final_result = det_result if use_mrbi else zero_result

    total_runtime = time.perf_counter() - t0
    return HybridSolveResult(
        success=final_result.success,
        z_star=final_result.z_star,
        residual=final_result.residual,
        used_mrbi=use_mrbi,
        trigger_reason="+".join(trigger_reasons),
        zero_result=zero_result,
        final_result=final_result,
        mrbi_candidate=candidate,
        sigma_min_zero=s_zero,
        sigma_min_det=s_det,
        total_runtime_sec=total_runtime,
    )


# ---------------------------------------------------------------------
# Small convenience wrapper for the implicit tanh layer
# ---------------------------------------------------------------------

def hybrid_mrbi_solve_tanh_layer(
    layer: ImplicitTanhLayer,
    x: Array,
    *,
    mrbi_cfg: MRBIConfig | None = None,
    hybrid_cfg: HybridConfig | None = None,
    root_cfg: RootSolveConfig | None = None,
    rng: Optional[np.random.Generator] = None,
) -> HybridSolveResult:
    """
    Convenience wrapper specialized to the implicit tanh layer.
    """
    F, J = make_residual_and_jacobian(layer)
    return hybrid_mrbi_solve(
        F,
        J,
        x,
        dim=layer.d,
        mrbi_cfg=mrbi_cfg,
        hybrid_cfg=hybrid_cfg,
        root_cfg=root_cfg,
        rng=rng,
    )


__all__ = [
    "Array",
    "ResidualFn",
    "JacobianFn",
    "ImplicitTanhLayer",
    "make_residual_and_jacobian",
    "RootSolveConfig",
    "MRBIConfig",
    "HybridConfig",
    "RootResult",
    "MRBICandidate",
    "HybridSolveResult",
    "solve_root",
    "jacobian_health",
    "sigma_min_jacobian",
    "MRBIOptimizer",
    "hybrid_mrbi_solve",
    "hybrid_mrbi_solve_tanh_layer",
]