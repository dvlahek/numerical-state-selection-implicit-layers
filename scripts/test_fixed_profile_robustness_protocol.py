"""Check the fixed-profile robustness protocol without running QNN training."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
sys.path.insert(0,str(ROOT/"scripts"))

import main_qnn_benchmark as bench
import run_locked_selected_profile_confirmation as confirm
from fixed_profile_robustness_common import (
    ANCHOR_RHO,
    RHO_VALUES,
    layer_at_rho,
    make_base_layer,
    prepare_dataset,
    spectral_radius,
)
from run_fixed_profile_rho_sensitivity import json_safe


def main():
    lock=confirm.load_lock()
    if tuple(lock["confirmation_plan"]["seeds"])!=tuple(range(40,50)):
        raise AssertionError("Confirmation seeds changed")
    if len(lock["selected_profiles"])!=9:
        raise AssertionError("Expected nine fixed dataset-specific methods")

    dataset=lock["development_evidence"]["datasets"][0]
    seed=40
    cfg,Xtr_p,_Xte_p,_ytr,_yte=prepare_dataset(
        dataset,seed,lock
    )

    base,base_seed=make_base_layer(Xtr_p,cfg)
    old_seed=(
        cfg.seed+1000+cfg.latent_dim+int(100*cfg.spectral_radius)
    )
    old=bench.make_random_implicit_layer(
        dx=Xtr_p.shape[1],
        d=cfg.latent_dim,
        cfg=cfg,
        seed=old_seed,
    )
    if base_seed!=old_seed:
        raise AssertionError("Anchor layer seed changed")
    for name in ("W","U","b"):
        if not np.array_equal(getattr(base,name),getattr(old,name)):
            raise AssertionError(
                f"rho=2.0 base layer differs in {name}"
            )

    for rho in RHO_VALUES:
        layer,base_rho,actual=layer_at_rho(base,rho)
        if not np.array_equal(layer.U,base.U):
            raise AssertionError(f"U changed at rho={rho}")
        if not np.array_equal(layer.b,base.b):
            raise AssertionError(f"b changed at rho={rho}")
        if not np.isclose(
            spectral_radius(layer.W),rho,atol=1e-10,rtol=0.0
        ):
            raise AssertionError(f"Spectral radius mismatch at rho={rho}")
        if rho==ANCHOR_RHO and not np.array_equal(layer.W,base.W):
            raise AssertionError("rho=2.0 W is not bitwise identical")
        if not np.isclose(base_rho,ANCHOR_RHO,atol=1e-10,rtol=0.0):
            raise AssertionError("Unexpected base spectral radius")

    cleaned=json_safe({
        "finite":1.25,
        "nan":float("nan"),
        "pos_inf":float("inf"),
        "neg_inf":float("-inf"),
        "nested":[np.float64("nan"),2.0],
    })
    expected={
        "finite":1.25,
        "nan":None,
        "pos_inf":None,
        "neg_inf":None,
        "nested":[None,2.0],
    }
    if cleaned!=expected:
        raise AssertionError(f"JSON-safe conversion drift: {cleaned}")

    print(
        "FIXED_PROFILE_ROBUSTNESS_PROTOCOL_TEST_OK "
        f"dataset={dataset} seed={seed} base_seed={base_seed} "
        f"rhos={','.join(f'{r:.2f}' for r in RHO_VALUES)} "
        "json_nonfinite_to_null=true"
    )


if __name__=="__main__":
    main()
