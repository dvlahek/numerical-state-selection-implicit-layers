"""Shared helpers for the fixed-profile robustness checks."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
sys.path.insert(0,str(ROOT/"scripts"))

import main_qnn_benchmark as bench
import mrbi
import run_locked_selected_profile_confirmation as confirm

RHO_VALUES=(1.20,1.60,2.00,2.25)
ANCHOR_RHO=2.00
RHO_TOL=1e-10


def spectral_radius(W):
    eigvals=np.linalg.eigvals(np.asarray(W,dtype=np.float64))
    return float(np.max(np.abs(eigvals)))


def rho_tag(rho):
    return f"{float(rho):.2f}".replace(".","p")


def make_base_layer(Xtr_p,cfg):
    """Recreate the exact confirmation layer at rho=2.0.

    The seed intentionally does not depend on the target sensitivity rho.
    """
    cfg_anchor=replace(cfg,spectral_radius=ANCHOR_RHO)
    seed=(
        cfg.seed+1000+cfg.latent_dim+int(100*ANCHOR_RHO)
    )
    layer=bench.make_random_implicit_layer(
        dx=Xtr_p.shape[1],
        d=cfg.latent_dim,
        cfg=cfg_anchor,
        seed=seed,
    )
    actual=spectral_radius(layer.W)
    if not np.isclose(actual,ANCHOR_RHO,atol=RHO_TOL,rtol=0.0):
        raise RuntimeError(
            f"Base layer spectral radius drift: {actual} != {ANCHOR_RHO}"
        )
    return layer,seed


def layer_at_rho(base_layer,target_rho):
    target=float(target_rho)
    if target not in RHO_VALUES:
        raise ValueError(f"rho={target} is outside the fixed protocol")
    base_rho=spectral_radius(base_layer.W)
    if target==ANCHOR_RHO:
        W=np.asarray(base_layer.W,dtype=np.float64).copy()
    else:
        W=np.asarray(base_layer.W,dtype=np.float64)*(target/base_rho)
    layer=mrbi.ImplicitTanhLayer(
        W=W,
        U=np.asarray(base_layer.U,dtype=np.float64).copy(),
        b=np.asarray(base_layer.b,dtype=np.float64).copy(),
    )
    actual=spectral_radius(layer.W)
    if not np.isclose(actual,target,atol=RHO_TOL,rtol=0.0):
        raise RuntimeError(f"Target spectral radius mismatch: {actual} != {target}")
    return layer,base_rho,actual


def prepare_dataset(dataset,seed,lock):
    cfg=confirm.make_cfg(seed,lock)
    X,y=bench.load_dataset(dataset,cfg)
    Xtr_raw,Xte_raw,ytr,yte=train_test_split(
        X,y,
        test_size=cfg.test_size,
        stratify=y,
        random_state=cfg.seed,
    )
    Xtr_p,Xte_p=bench.preprocess_to_pca(
        Xtr_raw,Xte_raw,cfg.pca_dim,cfg.seed,
    )
    return cfg,Xtr_p,Xte_p,ytr,yte


def selected_success_rate(kind,stats):
    if kind=="forced":
        return float(stats["forced_success_rate"])
    return float(stats["hybrid_success_rate"])
