"""
Benchmark MRBI-based implicit representations with simulated QNN readouts.

The zero-initialized solver is often sufficient in easier regimes. This
benchmark therefore includes non-contractive settings in which initialization
has a measurable effect on root finding.

It evaluates:
    1) PCA + classical/QNN readouts
    2) zero-initialized implicit features
    3) hybrid MRBI features
    4) forced MRBI features
    5) more aggressive hybrid trigger settings
    6) MRBI ablations: full, no_detector, residual_only

The benchmark supports several implicit-layer difficulties and writes raw and
summary CSV files. QNN evaluation is optional because all circuits are simulated
classically.

Install:
    pip install numpy scipy scikit-learn pandas torch pennylane

Requirement:
    mrbi.py must be in the same folder as this script.

Fast hard test:
    python main_qnn_benchmark.py --mode hard_quick --no-qnn

Full hard test:
    python main_qnn_benchmark.py --mode hard_article --no-qnn --seeds 0 1 2 3 4

Smaller QNN run:
    python main_qnn_benchmark.py --mode hard_quick --seeds 0 1 --qubits 4 6

All QNN results are classical simulations; no quantum-hardware or
quantum-advantage claim is made here.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import mrbi

warnings.filterwarnings("ignore", category=ConvergenceWarning)

IMPLEMENTATION_VERSION = "mrbi_continuation_v1"


# ---------------------------------------------------------------------
# Optional QNN dependency
# ---------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import pennylane as qml
    HAS_QNN = True
except Exception as e:
    HAS_QNN = False
    QNN_IMPORT_ERROR = repr(e)


# ---------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    seed: int = 42
    test_size: float = 0.30

    n_qubits: int = 4
    pca_dim: int = 4
    latent_dim: int = 16
    max_samples_per_class: int = 100

    # Hard implicit layer parameters
    spectral_radius: float = 1.75
    input_scale: float = 1.10
    bias_scale: float = 0.10
    hard_layer: bool = True

    # QNN
    run_qnn: bool = True
    qnn_epochs: int = 50
    qnn_lr: float = 0.01
    qnn_batch_size: int = 16
    qnn_layers: int = 2

    # Root solve
    root_method: str = "hybr"
    root_tol: float = 1e-10
    success_residual_tol: float = 1e-8

    # MRBI profile parameters
    profile_name: str = "full_balanced"
    sigmas: Tuple[float, ...] = (0.70, 0.25, 0.08, 0.02)
    mc_samples: int = 10
    maxiter_per_scale: int = 60
    refinement_iters: int = 60
    gamma: float = 0.04
    beta: float = 0.02
    lambda_zero: float = 0.04
    lambda_relative: float = 0.50
    search_radius: float = 1.50
    use_detector: bool = True
    use_newton_term: bool = True
    repeat_final_sigma: bool = False  # False preserves corrected continuation

    # Hybrid config
    hybrid_name: str = "standard"
    tau_r: float = 1e-8
    tau_sigma: float = 0.08
    residual_trigger_scale: float = 0.02
    accept_factor_vs_zero: float = 1.05

    # Forced MRBI acceptance mode
    forced_accept_if_better_residual: bool = True


@dataclass
class MRBIProfile:
    name: str
    sigmas: Tuple[float, ...]
    mc_samples: int
    maxiter_per_scale: int
    refinement_iters: int
    gamma: float
    beta: float
    lambda_zero: float
    lambda_relative: float
    search_radius: float
    use_detector: bool
    use_newton_term: bool


@dataclass
class HybridProfile:
    name: str
    tau_r: float
    tau_sigma: float
    residual_trigger_scale: float
    accept_factor_vs_zero: float


def get_mrbi_profiles(mode: str) -> List[MRBIProfile]:
    base = [
        MRBIProfile(
            name="full_light",
            sigmas=(0.45, 0.12, 0.03),
            mc_samples=6,
            maxiter_per_scale=30,
            refinement_iters=30,
            gamma=0.03,
            beta=0.01,
            lambda_zero=0.05,
            lambda_relative=0.75,
            search_radius=1.25,
            use_detector=True,
            use_newton_term=True,
        ),
        MRBIProfile(
            name="full_balanced",
            sigmas=(0.70, 0.25, 0.08, 0.02),
            mc_samples=10,
            maxiter_per_scale=60,
            refinement_iters=60,
            gamma=0.04,
            beta=0.02,
            lambda_zero=0.04,
            lambda_relative=0.50,
            search_radius=1.50,
            use_detector=True,
            use_newton_term=True,
        ),
        MRBIProfile(
            name="qnn_oriented",
            sigmas=(1.00, 0.50, 0.20, 0.07, 0.02),
            mc_samples=16,
            maxiter_per_scale=90,
            refinement_iters=90,
            gamma=0.08,
            beta=0.01,
            lambda_zero=0.01,
            lambda_relative=0.20,
            search_radius=2.00,
            use_detector=True,
            use_newton_term=True,
        ),
        MRBIProfile(
            name="qnn_exploratory",
            sigmas=(1.20, 0.60, 0.25, 0.10, 0.03),
            mc_samples=20,
            maxiter_per_scale=100,
            refinement_iters=100,
            gamma=0.12,
            beta=0.005,
            lambda_zero=0.005,
            lambda_relative=0.10,
            search_radius=2.50,
            use_detector=True,
            use_newton_term=True,
        ),
        MRBIProfile(
            name="no_detector",
            sigmas=(0.70, 0.25, 0.08, 0.02),
            mc_samples=1,
            maxiter_per_scale=60,
            refinement_iters=60,
            gamma=0.0,
            beta=0.02,
            lambda_zero=0.04,
            lambda_relative=0.50,
            search_radius=1.50,
            use_detector=False,
            use_newton_term=True,
        ),
        MRBIProfile(
            name="residual_only",
            sigmas=(0.70, 0.25, 0.08, 0.02),
            mc_samples=1,
            maxiter_per_scale=60,
            refinement_iters=60,
            gamma=0.0,
            beta=0.0,
            lambda_zero=0.04,
            lambda_relative=0.50,
            search_radius=1.50,
            use_detector=False,
            use_newton_term=False,
        ),
    ]

    if mode == "hard_article":
        base.append(
            MRBIProfile(
                name="full_strong",
                sigmas=(1.00, 0.50, 0.20, 0.07, 0.02),
                mc_samples=14,
                maxiter_per_scale=90,
                refinement_iters=90,
                gamma=0.05,
                beta=0.02,
                lambda_zero=0.03,
                lambda_relative=0.35,
                search_radius=2.00,
                use_detector=True,
                use_newton_term=True,
            )
        )

    return base


def get_hybrid_profiles(mode: str) -> List[HybridProfile]:
    """
    standard:
        Original conservative behavior.
    aggressive:
        Triggers more often using conditioning and relative residual.
    always_trigger:
        Runs MRBI for every sample but keeps the same acceptance rule.
        This isolates the effect of the MRBI candidate when Zero also succeeds.
    """
    profiles = [
        HybridProfile(
            name="standard",
            tau_r=1e-8,
            tau_sigma=0.08,
            residual_trigger_scale=0.02,
            accept_factor_vs_zero=1.05,
        ),
        HybridProfile(
            name="aggressive",
            tau_r=1e-12,
            tau_sigma=0.20,
            residual_trigger_scale=0.001,
            accept_factor_vs_zero=1.20,
        ),
        HybridProfile(
            name="always_trigger",
            tau_r=0.0,
            tau_sigma=1e9,
            residual_trigger_scale=0.0,
            accept_factor_vs_zero=1.20,
        ),
        HybridProfile(
            name="qnn_lenient",
            tau_r=0.0,
            tau_sigma=1e9,
            residual_trigger_scale=0.0,
            accept_factor_vs_zero=1.50,
        ),
    ]

    if mode == "hard_quick":
        return profiles

    profiles.append(
        HybridProfile(
            name="very_aggressive",
            tau_r=0.0,
            tau_sigma=1e9,
            residual_trigger_scale=0.0,
            accept_factor_vs_zero=2.00,
        )
    )
    return profiles


def apply_mrbi_profile(cfg: ExperimentConfig, profile: MRBIProfile) -> ExperimentConfig:
    return replace(
        cfg,
        profile_name=profile.name,
        sigmas=profile.sigmas,
        mc_samples=profile.mc_samples,
        maxiter_per_scale=profile.maxiter_per_scale,
        refinement_iters=profile.refinement_iters,
        gamma=profile.gamma,
        beta=profile.beta,
        lambda_zero=profile.lambda_zero,
        lambda_relative=profile.lambda_relative,
        search_radius=profile.search_radius,
        use_detector=profile.use_detector,
        use_newton_term=profile.use_newton_term,
    )


def apply_hybrid_profile(cfg: ExperimentConfig, profile: HybridProfile) -> ExperimentConfig:
    return replace(
        cfg,
        hybrid_name=profile.name,
        tau_r=profile.tau_r,
        tau_sigma=profile.tau_sigma,
        residual_trigger_scale=profile.residual_trigger_scale,
        accept_factor_vs_zero=profile.accept_factor_vs_zero,
    )


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def balanced_subsample(X: np.ndarray, y: np.ndarray, max_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    keep: List[int] = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        keep.extend(idx[:max_per_class].tolist())
    keep = np.array(keep, dtype=int)
    rng.shuffle(keep)
    return X[keep], y[keep]


def load_dataset(name: str, cfg: ExperimentConfig):
    name = name.lower()

    if name == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data.astype(np.float64), data.target.astype(int)

    elif name == "wine_binary":
        data = load_wine()
        X = data.data.astype(np.float64)
        yr = data.target.astype(int)
        mask = yr < 2
        X, y = X[mask], yr[mask].astype(int)

    elif name == "iris_binary":
        data = load_iris()
        X = data.data.astype(np.float64)
        yr = data.target.astype(int)
        mask = yr > 0
        X, y = X[mask], (yr[mask] == 2).astype(int)

    elif name == "digits_3_vs_8":
        data = load_digits()
        X = data.data.astype(np.float64) / 16.0
        yr = data.target.astype(int)
        mask = (yr == 3) | (yr == 8)
        X, y = X[mask], (yr[mask] == 8).astype(int)

    elif name == "digits_1_vs_7":
        data = load_digits()
        X = data.data.astype(np.float64) / 16.0
        yr = data.target.astype(int)
        mask = (yr == 1) | (yr == 7)
        X, y = X[mask], (yr[mask] == 7).astype(int)

    elif name == "digits_4_vs_9":
        data = load_digits()
        X = data.data.astype(np.float64) / 16.0
        yr = data.target.astype(int)
        mask = (yr == 4) | (yr == 9)
        X, y = X[mask], (yr[mask] == 9).astype(int)
    elif name == "digits_5_vs_6":
        data = load_digits()
        X = data.data.astype(np.float64) / 16.0
        yr = data.target.astype(int)
        mask = (yr == 5) | (yr == 6)
        X, y = X[mask], (yr[mask] == 6).astype(int)

    elif name == "digits_2_vs_7":
        data = load_digits()
        X = data.data.astype(np.float64) / 16.0
        yr = data.target.astype(int)
        mask = (yr == 2) | (yr == 7)
        X, y = X[mask], (yr[mask] == 7).astype(int)

    elif name == "wine_1_vs_2":
        data = load_wine()
        X = data.data.astype(np.float64)
        yr = data.target.astype(int)
        mask = (yr == 1) | (yr == 2)
        X, y = X[mask], (yr[mask] == 2).astype(int)

    elif name == "wine_0_vs_2":
        data = load_wine()
        X = data.data.astype(np.float64)
        yr = data.target.astype(int)
        mask = (yr == 0) | (yr == 2)
        X, y = X[mask], (yr[mask] == 2).astype(int)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return balanced_subsample(X, y, cfg.max_samples_per_class, cfg.seed)


def preprocess_to_pca(X_train, X_test, pca_dim: int, seed: int):
    n_components = min(pca_dim, X_train.shape[1], max(1, X_train.shape[0] - 1))
    pipe = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, random_state=seed),
        StandardScaler(),
    )
    Xtr = pipe.fit_transform(X_train).astype(np.float64)
    Xte = pipe.transform(X_test).astype(np.float64)

    if n_components < pca_dim:
        Xtr = np.hstack([Xtr, np.zeros((Xtr.shape[0], pca_dim - n_components))])
        Xte = np.hstack([Xte, np.zeros((Xte.shape[0], pca_dim - n_components))])

    return np.tanh(Xtr), np.tanh(Xte)


# ---------------------------------------------------------------------
# Metrics and readouts
# ---------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_score=None):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score)) if y_score is not None else np.nan
    except Exception:
        out["roc_auc"] = np.nan
    return out


def train_logreg(X_train, y_train, X_test, y_test, seed):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed))
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    pred = clf.predict(X_test)
    score = clf.predict_proba(X_test)[:, 1]
    m = compute_metrics(y_test, pred, score)
    m["train_time_sec"] = train_time
    return m


def train_mlp(X_train, y_train, X_test, y_test, seed):
    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            alpha=1e-4,
            max_iter=3000,
            random_state=seed,
            early_stopping=False,
        ),
    )
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    pred = clf.predict(X_test)
    score = clf.predict_proba(X_test)[:, 1]
    m = compute_metrics(y_test, pred, score)
    m["train_time_sec"] = train_time
    return m


def train_svm_rbf(X_train, y_train, X_test, y_test, seed):
    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed),
    )
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    pred = clf.predict(X_test)
    score = clf.predict_proba(X_test)[:, 1]
    m = compute_metrics(y_test, pred, score)
    m["train_time_sec"] = train_time
    return m


def train_rf(X_train, y_train, X_test, y_test, seed):
    clf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=1,
    )
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    pred = clf.predict(X_test)
    score = clf.predict_proba(X_test)[:, 1]
    m = compute_metrics(y_test, pred, score)
    m["train_time_sec"] = train_time
    return m


def train_gb(X_train, y_train, X_test, y_test, seed):
    clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=seed)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    pred = clf.predict(X_test)
    score = clf.predict_proba(X_test)[:, 1]
    m = compute_metrics(y_test, pred, score)
    m["train_time_sec"] = train_time
    return m


READOUTS = {
    "logreg": train_logreg,
    "mlp": train_mlp,
    "svm_rbf": train_svm_rbf,
    "rf": train_rf,
    "gb": train_gb,
}


# ---------------------------------------------------------------------
# QNN
# ---------------------------------------------------------------------

if HAS_QNN:
    class TorchQNN(nn.Module):
        def __init__(self, n_features: int, n_qubits: int, n_layers: int, seed: int):
            super().__init__()
            torch.manual_seed(seed)

            self.pre = nn.Linear(n_features, n_qubits)
            self.post = nn.Linear(n_qubits, 2)
            self.n_qubits = n_qubits

            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def circuit(inputs, weights):
                for q in range(n_qubits):
                    qml.RY(inputs[q], wires=q)

                for l in range(n_layers):
                    for q in range(n_qubits):
                        qml.RY(weights[l, q, 0], wires=q)
                        qml.RZ(weights[l, q, 1], wires=q)
                    for q in range(n_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])
                    if n_qubits > 2:
                        qml.CNOT(wires=[n_qubits - 1, 0])

                return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

            self.circuit = circuit
            self.q_weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 2))

        def forward(self, x):
            angles = torch.tanh(self.pre(x)) * np.pi
            outs = []
            for i in range(angles.shape[0]):
                vals = self.circuit(angles[i], self.q_weights)
                vals = torch.stack(vals).float()
                outs.append(vals)
            return self.post(torch.stack(outs, dim=0))


def train_qnn(X_train, y_train, X_test, y_test, cfg: ExperimentConfig, seed: int):
    if not cfg.run_qnn:
        return {"accuracy": np.nan, "balanced_accuracy": np.nan, "f1": np.nan, "roc_auc": np.nan, "train_time_sec": 0.0, "note": "QNN disabled"}

    if not HAS_QNN:
        return {"accuracy": np.nan, "balanced_accuracy": np.nan, "f1": np.nan, "roc_auc": np.nan, "train_time_sec": 0.0, "note": f"QNN import failed: {QNN_IMPORT_ERROR}"}

    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xte = torch.tensor(X_test, dtype=torch.float32)

    model = TorchQNN(X_train.shape[1], cfg.n_qubits, cfg.qnn_layers, seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.qnn_lr)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    for _epoch in range(cfg.qnn_epochs):
        perm = rng.permutation(len(Xtr))
        for start in range(0, len(Xtr), cfg.qnn_batch_size):
            idx = perm[start:start + cfg.qnn_batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()

    train_time = time.perf_counter() - t0

    with torch.no_grad():
        probs = torch.softmax(model(Xte), dim=1).numpy()
        pred = np.argmax(probs, axis=1)

    m = compute_metrics(y_test, pred, probs[:, 1])
    m["train_time_sec"] = train_time
    return m


# ---------------------------------------------------------------------
# Hard implicit layer
# ---------------------------------------------------------------------

def make_random_implicit_layer(dx: int, d: int, cfg: ExperimentConfig, seed: int):
    """
    Construct W with requested spectral radius.

    hard_layer=True creates a mildly ill-conditioned non-normal matrix by
    combining a random orthogonal basis with log-spaced singular-like scales.
    This often makes zero solves more fragile and makes MRBI triggering meaningful.
    """
    rng = np.random.default_rng(seed)

    if cfg.hard_layer:
        A = rng.normal(size=(d, d))
        Q, _ = np.linalg.qr(A)

        B = rng.normal(size=(d, d))
        R, _ = np.linalg.qr(B)

        scales = np.geomspace(1.0, 0.08, d)
        W = Q @ np.diag(scales) @ R.T

        # Add non-normal low-rank skew/upper component.
        upper = np.triu(rng.normal(size=(d, d)), k=1)
        W = W + 0.35 * upper / max(np.linalg.norm(upper, 2), 1e-12)
    else:
        W = rng.normal(size=(d, d))

    eigvals = np.linalg.eigvals(W)
    radius = float(np.max(np.abs(eigvals)))
    W = W / max(radius, 1e-12) * cfg.spectral_radius

    U = rng.normal(scale=cfg.input_scale / np.sqrt(max(dx, 1)), size=(d, dx))
    b = rng.normal(scale=cfg.bias_scale, size=d)

    return mrbi.ImplicitTanhLayer(W=W, U=U, b=b)


def make_root_cfg(cfg: ExperimentConfig):
    return mrbi.RootSolveConfig(
        method=cfg.root_method,
        tol=cfg.root_tol,
        success_residual_tol=cfg.success_residual_tol,
    )


def make_mrbi_cfg(cfg: ExperimentConfig):
    return mrbi.MRBIConfig(
        sigmas=cfg.sigmas,
        alpha=1.0,
        beta=cfg.beta,
        gamma=cfg.gamma,
        lambda_zero=cfg.lambda_zero,
        lambda_relative=cfg.lambda_relative,
        epsilon=1e-8,
        search_radius=cfg.search_radius,
        mc_samples=cfg.mc_samples,
        maxiter_per_scale=cfg.maxiter_per_scale,
        refinement_iters=cfg.refinement_iters,
        use_continuation=True,
        repeat_final_sigma=cfg.repeat_final_sigma,
        use_refinement=True,
        use_detector=cfg.use_detector,
        use_newton_term=cfg.use_newton_term,
        n_starts=1,
        random_start_scale=0.05,
        newton_damping=1e-6,
        optimizer_ftol=1e-10,
        use_fixed_probes_per_scale=True,
        use_antithetic_sampling=True,
    )


def make_hybrid_cfg(cfg: ExperimentConfig):
    return mrbi.HybridConfig(
        tau_r=cfg.tau_r,
        tau_sigma=cfg.tau_sigma,
        accept_factor_vs_zero=cfg.accept_factor_vs_zero,
        use_relative_residual_trigger=True,
        residual_trigger_scale=cfg.residual_trigger_scale,
        trigger_on_solver_failure=True,
        trigger_on_bad_jacobian=True,
        tie_residual_rtol=0.02,
        prefer_better_conditioning_on_tie=True,
        accept_any_finite_success_if_zero_failed=True,
    )


# ---------------------------------------------------------------------
# Feature solvers: zero, hybrid, forced
# ---------------------------------------------------------------------

def solve_zero_features(X, layer, cfg: ExperimentConfig):
    F, J = mrbi.make_residual_and_jacobian(layer)
    root_cfg = make_root_cfg(cfg)

    Z, residuals, success, nfev, runtimes, smins = [], [], [], [], [], []
    z0 = np.zeros(layer.d, dtype=np.float64)

    for x in X:
        res = mrbi.solve_root(F, J, x, z0, cfg=root_cfg)
        smin, _ok = mrbi.jacobian_health(J, x, res.z_star)

        Z.append(res.z_star)
        residuals.append(float(res.residual))
        success.append(int(res.success))
        nfev.append(int(res.nfev))
        runtimes.append(float(res.runtime_sec))
        smins.append(float(smin))

    stats = {
        "zero_success_rate": float(np.mean(success)),
        "zero_mean_residual": float(np.mean(residuals)),
        "zero_median_residual": float(np.median(residuals)),
        "zero_mean_nfev": float(np.mean(nfev)),
        "zero_mean_runtime_sec_per_sample": float(np.mean(runtimes)),
        "zero_mean_sigma_min": float(np.mean(smins)),
    }
    return np.asarray(Z), stats


def solve_hybrid_features(X, layer, cfg: ExperimentConfig, seed: int):
    root_cfg = make_root_cfg(cfg)
    mrbi_cfg = make_mrbi_cfg(cfg)
    hybrid_cfg = make_hybrid_cfg(cfg)
    rng = np.random.default_rng(seed)

    Z = []
    success = []
    residuals = []
    used_mrbi = []
    triggered = []
    nfev_zero = []
    nfev_final = []
    runtimes = []
    s_zero = []
    s_det = []
    reasons: Dict[str, int] = {}

    for x in X:
        res = mrbi.hybrid_mrbi_solve_tanh_layer(
            layer,
            x,
            mrbi_cfg=mrbi_cfg,
            hybrid_cfg=hybrid_cfg,
            root_cfg=root_cfg,
            rng=rng,
        )
        Z.append(res.z_star)
        success.append(int(res.success))
        residuals.append(float(res.residual))
        used_mrbi.append(int(res.used_mrbi))
        triggered.append(int(res.trigger_reason != "none"))
        nfev_zero.append(int(res.zero_result.nfev))
        nfev_final.append(int(res.final_result.nfev))
        runtimes.append(float(res.total_runtime_sec or 0.0))
        s_zero.append(float(res.sigma_min_zero or 0.0))
        s_det.append(float(res.sigma_min_det) if res.sigma_min_det is not None else np.nan)
        reasons[res.trigger_reason] = reasons.get(res.trigger_reason, 0) + 1

    stats = {
        "hybrid_success_rate": float(np.mean(success)),
        "hybrid_trigger_rate": float(np.mean(triggered)),
        "hybrid_accept_rate": float(np.mean(used_mrbi)),
        "hybrid_mean_residual": float(np.mean(residuals)),
        "hybrid_median_residual": float(np.median(residuals)),
        "hybrid_mean_zero_nfev": float(np.mean(nfev_zero)),
        "hybrid_mean_final_nfev": float(np.mean(nfev_final)),
        "hybrid_mean_runtime_sec_per_sample": float(np.mean(runtimes)),
        "hybrid_mean_sigma_min_zero": float(np.mean(s_zero)),
        "hybrid_mean_sigma_min_det": float(np.nanmean(s_det)),
        "hybrid_trigger_reasons_json": json.dumps(reasons),
    }
    return np.asarray(Z), stats


def solve_forced_mrbi_features(X, layer, cfg: ExperimentConfig, seed: int):
    """
    Forced MRBI mode.

    Unlike hybrid_mrbi_solve_tanh_layer, this always builds an MRBI candidate and
    solves from it. The final accepted z is:
        - MRBI result if it has better residual than zero, or if zero failed
        - otherwise zero result
    if forced_accept_if_better_residual=True.

    This mode is used as an ablation of MRBI candidate construction. The hybrid
    trigger remains the default policy for normal benchmark comparisons.
    """
    F, J = mrbi.make_residual_and_jacobian(layer)
    root_cfg = make_root_cfg(cfg)
    mrbi_cfg = make_mrbi_cfg(cfg)
    rng = np.random.default_rng(seed)

    Z = []
    chose_mrbi = []
    success = []
    zero_success = []
    mrbi_success = []
    zero_residuals = []
    mrbi_residuals = []
    final_residuals = []
    zero_nfev = []
    mrbi_nfev = []
    obj_calls = []
    runtimes = []
    s_zero = []
    s_mrbi = []

    z0 = np.zeros(layer.d, dtype=np.float64)

    for x in X:
        t0 = time.perf_counter()

        zero = mrbi.solve_root(F, J, x, z0, cfg=root_cfg)
        s0, _ = mrbi.jacobian_health(J, x, zero.z_star)

        opt = mrbi.MRBIOptimizer(
            F=F,
            J=J,
            x=x,
            dim=layer.d,
            config=mrbi_cfg,
            rng=rng,
        )
        cand = opt.optimize()
        det = mrbi.solve_root(F, J, x, cand.z_init, cfg=root_cfg)
        sd, _ = mrbi.jacobian_health(J, x, det.z_star)

        if cfg.forced_accept_if_better_residual:
            use_mrbi = (
                (not zero.success and det.success)
                or (np.isfinite(det.residual) and det.residual < zero.residual)
                or (
                    np.isfinite(det.residual)
                    and np.isfinite(zero.residual)
                    #and det.residual <= 1.02 * zero.residual
                    and det.residual <= 1.10 * zero.residual
                    and sd > s0
                )
            )
        else:
            use_mrbi = True

        final = det if use_mrbi else zero

        Z.append(final.z_star)
        chose_mrbi.append(int(use_mrbi))
        success.append(int(final.success))
        zero_success.append(int(zero.success))
        mrbi_success.append(int(det.success))
        zero_residuals.append(float(zero.residual))
        mrbi_residuals.append(float(det.residual))
        final_residuals.append(float(final.residual))
        zero_nfev.append(int(zero.nfev))
        mrbi_nfev.append(int(det.nfev))
        obj_calls.append(int(cand.n_objective_calls))
        runtimes.append(float(time.perf_counter() - t0))
        s_zero.append(float(s0))
        s_mrbi.append(float(sd))

    stats = {
        "forced_success_rate": float(np.mean(success)),
        "forced_choose_mrbi_rate": float(np.mean(chose_mrbi)),
        "forced_zero_success_rate": float(np.mean(zero_success)),
        "forced_mrbi_success_rate": float(np.mean(mrbi_success)),
        "forced_zero_mean_residual": float(np.mean(zero_residuals)),
        "forced_mrbi_mean_residual": float(np.mean(mrbi_residuals)),
        "forced_final_mean_residual": float(np.mean(final_residuals)),
        "forced_zero_median_residual": float(np.median(zero_residuals)),
        "forced_mrbi_median_residual": float(np.median(mrbi_residuals)),
        "forced_zero_mean_nfev": float(np.mean(zero_nfev)),
        "forced_mrbi_mean_nfev": float(np.mean(mrbi_nfev)),
        "forced_mean_objective_calls": float(np.mean(obj_calls)),
        "forced_mean_runtime_sec_per_sample": float(np.mean(runtimes)),
        "forced_mean_sigma_min_zero": float(np.mean(s_zero)),
        "forced_mean_sigma_min_mrbi": float(np.mean(s_mrbi)),
    }
    return np.asarray(Z), stats


def standardize_pair(Xtr, Xte):
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xte)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def add_metadata(row: Dict[str, Any], dataset: str, method: str, representation: str, readout: str, cfg: ExperimentConfig):
    row.update({
        "dataset": dataset,
        "implementation_version": IMPLEMENTATION_VERSION,
        "method": method,
        "representation": representation,
        "readout": readout,
        "seed": cfg.seed,
        "n_qubits": cfg.n_qubits,
        "pca_dim": cfg.pca_dim,
        "latent_dim": cfg.latent_dim,
        "spectral_radius": cfg.spectral_radius,
        "input_scale": cfg.input_scale,
        "hard_layer": cfg.hard_layer,
        "profile_name": cfg.profile_name,
        "hybrid_name": cfg.hybrid_name,
        "max_samples_per_class": cfg.max_samples_per_class,
    })
    return row


def evaluate_readouts(dataset, representation, Xtr, Xte, ytr, yte, cfg, readouts, extra_stats=None):
    rows = []
    extra_stats = extra_stats or {}

    for readout in readouts:
        if readout == "qnn":
            metrics = train_qnn(Xtr, ytr, Xte, yte, cfg, cfg.seed + 777)
        else:
            metrics = READOUTS[readout](Xtr, ytr, Xte, yte, cfg.seed)

        method = f"{representation}_{readout}"
        row = {**metrics, **{f"stat_{k}": v for k, v in extra_stats.items()}}
        rows.append(add_metadata(row, dataset, method, representation, readout, cfg))

    return rows


def run_job(dataset: str, cfg: ExperimentConfig, mrbi_profiles: List[MRBIProfile], hybrid_profiles: List[HybridProfile], readouts_pca: List[str], readouts_impl: List[str]):
    X, y = load_dataset(dataset, cfg)

    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        stratify=y,
        random_state=cfg.seed,
    )

    Xtr_p, Xte_p = preprocess_to_pca(Xtr_raw, Xte_raw, cfg.pca_dim, cfg.seed)

    rows: List[Dict[str, Any]] = []

    print(f"[{dataset} seed={cfg.seed}] PCA readouts", flush=True)
    rows.extend(evaluate_readouts(dataset, "pca", Xtr_p, Xte_p, ytr, yte, cfg, readouts_pca))

    layer = make_random_implicit_layer(
        dx=Xtr_p.shape[1],
        d=cfg.latent_dim,
        cfg=cfg,
        seed=cfg.seed + 1000 + cfg.latent_dim + int(100 * cfg.spectral_radius),
    )

    # Zero control.
    print(f"[{dataset} seed={cfg.seed}] Zero-init implicit features", flush=True)
    t0 = time.perf_counter()
    Ztr_zero, st_tr_zero = solve_zero_features(Xtr_p, layer, cfg)
    Zte_zero, st_te_zero = solve_zero_features(Xte_p, layer, cfg)
    zero_total_time = time.perf_counter() - t0
    Ztr_zero, Zte_zero = standardize_pair(Ztr_zero, Zte_zero)
    st = {f"test_{k}": v for k, v in st_te_zero.items()}
    st["feature_total_time_sec"] = zero_total_time
    rows.extend(evaluate_readouts(dataset, "implicit_zero", Ztr_zero, Zte_zero, ytr, yte, cfg, readouts_impl, st))

    # MRBI profiles and hybrid settings.
    for mp in mrbi_profiles:
        cfg_mp = apply_mrbi_profile(cfg, mp)
        print(f"[{dataset} seed={cfg.seed}] Forced MRBI profile: {mp.name}", flush=True)

        # Forced MRBI candidate ablation.
        t0 = time.perf_counter()
        Ztr_forced, st_tr_forced = solve_forced_mrbi_features(Xtr_p, layer, cfg_mp, cfg.seed + 2000)
        Zte_forced, st_te_forced = solve_forced_mrbi_features(Xte_p, layer, cfg_mp, cfg.seed + 3000)
        forced_total_time = time.perf_counter() - t0
        Ztr_forced, Zte_forced = standardize_pair(Ztr_forced, Zte_forced)
        st = {f"test_{k}": v for k, v in st_te_forced.items()}
        st["feature_total_time_sec"] = forced_total_time
        rows.extend(
            evaluate_readouts(
                dataset,
                f"forced_{mp.name}",
                Ztr_forced,
                Zte_forced,
                ytr,
                yte,
                cfg_mp,
                readouts_impl,
                st,
            )
        )

        # Hybrid variants.
        for hp in hybrid_profiles:
            cfg_h = apply_hybrid_profile(cfg_mp, hp)
            print(f"[{dataset} seed={cfg.seed}] Hybrid: {hp.name} / {mp.name}", flush=True)

            t0 = time.perf_counter()
            Ztr_h, st_tr_h = solve_hybrid_features(
                Xtr_p,
                layer,
                cfg_h,
                cfg.seed + 4000,
            )
            Zte_h, st_te_h = solve_hybrid_features(
                Xte_p,
                layer,
                cfg_h,
                cfg.seed + 5000,
            )
            hybrid_total_time = time.perf_counter() - t0

            Ztr_h, Zte_h = standardize_pair(Ztr_h, Zte_h)

            st = {f"test_{k}": v for k, v in st_te_h.items()}
            st["feature_total_time_sec"] = hybrid_total_time

            rows.extend(
                evaluate_readouts(
                    dataset,
                    f"hybrid_{hp.name}_{mp.name}",
                    Ztr_h,
                    Zte_h,
                    ytr,
                    yte,
                    cfg_h,
                    readouts_impl,
                    st,
                )
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def summarize_results(df: pd.DataFrame):
    group_cols = [
        "dataset",
        "method",
        "representation",
        "readout",
        "n_qubits",
        "latent_dim",
        "spectral_radius",
        "input_scale",
        "profile_name",
        "hybrid_name",
    ]

    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "train_time_sec",
        "stat_feature_total_time_sec",
        "stat_test_zero_success_rate",
        "stat_test_zero_mean_residual",
        "stat_test_zero_mean_nfev",
        "stat_test_zero_mean_sigma_min",
        "stat_test_hybrid_trigger_rate",
        "stat_test_hybrid_accept_rate",
        "stat_test_hybrid_mean_residual",
        "stat_test_hybrid_mean_runtime_sec_per_sample",
        "stat_test_forced_choose_mrbi_rate",
        "stat_test_forced_zero_mean_residual",
        "stat_test_forced_mrbi_mean_residual",
        "stat_test_forced_final_mean_residual",
        "stat_test_forced_zero_mean_nfev",
        "stat_test_forced_mrbi_mean_nfev",
        "stat_test_forced_mean_sigma_min_zero",
        "stat_test_forced_mean_sigma_min_mrbi",
    ]

    metric_cols = [c for c in metric_cols if c in df.columns]

    rows = []

    for keys, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_runs"] = int(len(g))

        for c in metric_cols:
            vals = pd.to_numeric(g[c], errors="coerce")
            row[f"{c}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"{c}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0

        rows.append(row)

    out = pd.DataFrame(rows)

    if "balanced_accuracy_mean" in out.columns:
        out = out.sort_values(
            [
                "dataset",
                "spectral_radius",
                "n_qubits",
                "balanced_accuracy_mean",
                "roc_auc_mean",
            ],
            ascending=[True, True, True, False, False],
        )

    return out


def print_leaderboard(summary: pd.DataFrame, top_k=10):
    cols = [
        "dataset",
        "spectral_radius",
        "n_qubits",
        "method",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "roc_auc_mean",
        "n_runs",
    ]
    cols = [c for c in cols if c in summary.columns]

    for (dataset, rho, nq), g in summary.groupby(["dataset", "spectral_radius", "n_qubits"], dropna=False):
        print(f"\n--- Leaderboard: dataset={dataset}, rho={rho}, qubits={nq} ---")
        print(g[cols].head(top_k).to_string(index=False))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MRBI-QNN benchmark.")
    p.add_argument("--mode", choices=["hard_quick", "hard_article"], default="hard_quick")
    p.add_argument("--datasets", nargs="+", default=[
        "breast_cancer", "wine_binary", "wine_0_vs_2", "wine_1_vs_2",
        "digits_1_vs_7", "digits_2_vs_7", "digits_3_vs_8",
        "digits_4_vs_9", "digits_5_vs_6",
    ])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--qubits", nargs="+", type=int, default=[4])
    p.add_argument("--latent-dims", nargs="+", type=int, default=[16])
    p.add_argument("--spectral-radii", nargs="+", type=float, default=[2.0])
    p.add_argument("--input-scales", nargs="+", type=float, default=[1.1])
    p.add_argument("--max-samples-per-class", type=int, default=80)
    p.add_argument("--qnn-epochs", type=int, default=60)
    p.add_argument("--qnn-layers", type=int, default=2)
    p.add_argument("--no-qnn", action="store_true")
    p.add_argument("--easy-layer", action="store_true",
                   help="Use the standard matrix construction instead of the hard regime.")
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--out-raw", default="article_qnn_continuation_v1_raw.csv")
    p.add_argument("--out-summary", default="article_qnn_continuation_v1_summary.csv")
    p.add_argument("--out-config", default="article_qnn_continuation_v1_config.json")
    return p.parse_args()


def run_job_worker(job_args):
    """
    Top-level worker function required for Windows multiprocessing.

    job_args contains:
        dataset, seed, nq, ld, rho, inscale,
        max_samples, qnn_epochs, qnn_layers, run_qnn, hard_layer,
        mode, readouts_pca, readouts_impl
    """
    (
        dataset,
        seed,
        nq,
        ld,
        rho,
        inscale,
        max_samples,
        qnn_epochs,
        qnn_layers,
        run_qnn,
        hard_layer,
        mode,
        readouts_pca,
        readouts_impl,
    ) = job_args

    cfg = ExperimentConfig(
        seed=seed,
        n_qubits=nq,
        pca_dim=nq,
        latent_dim=ld,
        max_samples_per_class=max_samples,
        spectral_radius=rho,
        input_scale=inscale,
        hard_layer=hard_layer,
        run_qnn=run_qnn,
        qnn_epochs=qnn_epochs,
        qnn_layers=qnn_layers,
    )

    mrbi_profiles = get_mrbi_profiles(mode)
    hybrid_profiles = get_hybrid_profiles(mode)

    df = run_job(
        dataset,
        cfg,
        mrbi_profiles,
        hybrid_profiles,
        readouts_pca,
        readouts_impl,
    )

    return {
        "dataset": dataset,
        "seed": seed,
        "n_qubits": nq,
        "latent_dim": ld,
        "spectral_radius": rho,
        "input_scale": inscale,
        "df": df,
    }



def main():
    args = parse_args()

    if args.mode == "hard_quick":
        datasets = args.datasets or ["breast_cancer", "wine_binary", "digits_3_vs_8"]
        seeds = args.seeds or [0]
        qubits = args.qubits or [4]
        latent_dims = args.latent_dims or [16]
        spectral_radii = args.spectral_radii or [1.60, 2.00]
        input_scales = args.input_scales or [1.10]
        max_samples = args.max_samples_per_class or 80
        qnn_epochs = args.qnn_epochs or 35
        readouts_pca = ["logreg", "mlp", "svm_rbf"] + ([] if args.no_qnn else ["qnn"])
        readouts_impl = ["logreg", "svm_rbf"] + ([] if args.no_qnn else ["qnn"])

    else:
        datasets = args.datasets or ["breast_cancer", "wine_binary", "digits_3_vs_8", "digits_1_vs_7", "digits_4_vs_9"]
        seeds = args.seeds or [0, 1, 2, 3, 4]
        qubits = args.qubits or [4, 6]
        latent_dims = args.latent_dims or [12, 16]
        spectral_radii = args.spectral_radii or [1.35, 1.60, 2.00, 2.25]
        input_scales = args.input_scales or [0.90, 1.20]
        max_samples = args.max_samples_per_class or 100
        qnn_epochs = args.qnn_epochs or 50
        readouts_pca = ["logreg", "mlp", "svm_rbf", "rf", "gb"] + ([] if args.no_qnn else ["qnn"])
        readouts_impl = ["logreg", "svm_rbf"] + ([] if args.no_qnn else ["qnn"])

    mrbi_profiles = get_mrbi_profiles(args.mode)
    hybrid_profiles = get_hybrid_profiles(args.mode)

    cfg_dump = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "mode": args.mode,
        "datasets": datasets,
        "seeds": seeds,
        "qubits": qubits,
        "latent_dims": latent_dims,
        "spectral_radii": spectral_radii,
        "input_scales": input_scales,
        "max_samples_per_class": max_samples,
        "qnn_epochs": qnn_epochs,
        "qnn_layers": args.qnn_layers,
        "run_qnn": not args.no_qnn,
        "hard_layer": not args.easy_layer,
        "readouts_pca": readouts_pca,
        "readouts_impl": readouts_impl,
        "mrbi_profiles": [p.__dict__ for p in mrbi_profiles],
        "hybrid_profiles": [p.__dict__ for p in hybrid_profiles],
    }

    print("Configuration:")
    print(json.dumps(cfg_dump, indent=2, default=str))
    Path(args.out_config).write_text(json.dumps(cfg_dump, indent=2, default=str), encoding="utf-8")

    job_args_list = []
    for dataset in datasets:
        for seed in seeds:
            for nq in qubits:
                for ld in latent_dims:
                    for rho in spectral_radii:
                        for inscale in input_scales:
                            job_args_list.append(
                                (
                                    dataset,
                                    seed,
                                    nq,
                                    ld,
                                    rho,
                                    inscale,
                                    max_samples,
                                    qnn_epochs,
                                    args.qnn_layers,
                                    not args.no_qnn,
                                    not args.easy_layer,
                                    args.mode,
                                    readouts_pca,
                                    readouts_impl,
                                )
                            )

    total = len(job_args_list)
    n_workers = max(1, int(getattr(args, "n_workers", 1)))
    n_workers = min(n_workers, total)

    print(f"\nTotal outer jobs: {total}")
    print(f"Parallel workers: {n_workers}")
    print("On Windows, use --n-workers 1 if multiprocessing causes problems.")

    all_results = []
    completed = 0

    if n_workers == 1:
        for ja in job_args_list:
            completed += 1
            dataset, seed, nq, ld, rho, inscale = ja[:6]
            print(
                f"\n[{completed}/{total}] dataset={dataset}, seed={seed}, "
                f"qubits={nq}, latent_dim={ld}, rho={rho}, input_scale={inscale}"
            )

            out = run_job_worker(ja)
            df = out["df"]
            all_results.append(df)

            raw = pd.concat(all_results, ignore_index=True)
            raw.to_csv(args.out_raw, index=False)

            show_cols = ["dataset", "method", "balanced_accuracy", "roc_auc", "stat_test_hybrid_trigger_rate", "stat_test_forced_choose_mrbi_rate"]
            show_cols = [c for c in show_cols if c in df.columns]
            print(df[show_cols].sort_values("balanced_accuracy", ascending=False).head(12).to_string(index=False))

    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(run_job_worker, ja) for ja in job_args_list]

            for fut in as_completed(futures):
                completed += 1
                try:
                    out = fut.result()
                except Exception as e:
                    print(f"\n[{completed}/{total}] JOB FAILED: {type(e).__name__}: {e}")
                    continue

                df = out["df"]
                all_results.append(df)

                print(
                    f"\n[{completed}/{total}] DONE dataset={out['dataset']}, "
                    f"seed={out['seed']}, qubits={out['n_qubits']}, "
                    f"latent_dim={out['latent_dim']}, rho={out['spectral_radius']}, "
                    f"input_scale={out['input_scale']}"
                )

                raw = pd.concat(all_results, ignore_index=True)
                raw.to_csv(args.out_raw, index=False)

                # Save partial summary too, so you can inspect progress during long runs.
                partial_summary = summarize_results(raw)
                partial_summary.to_csv(args.out_summary, index=False)

                show_cols = ["dataset", "method", "balanced_accuracy", "roc_auc", "stat_test_hybrid_trigger_rate", "stat_test_forced_choose_mrbi_rate"]
                show_cols = [c for c in show_cols if c in df.columns]
                print(df[show_cols].sort_values("balanced_accuracy", ascending=False).head(8).to_string(index=False))

    if not all_results:
        raise RuntimeError("No jobs completed successfully.")

    results = pd.concat(all_results, ignore_index=True)
    summary = summarize_results(results)

    results.to_csv(args.out_raw, index=False)
    summary.to_csv(args.out_summary, index=False)

    print(f"\nSaved raw results to: {args.out_raw}")
    print(f"Saved summary to: {args.out_summary}")
    print(f"Saved config to: {args.out_config}")

    print_leaderboard(summary)


if __name__ == "__main__":
    main()
