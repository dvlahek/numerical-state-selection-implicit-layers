"""Confirm MRBI profiles selected on development seeds 0-4.

The dataset-specific method map is fixed in:
    experiments/selected_profile_confirmation_lock_v1.json

Confirmation uses new seeds 40-49. For each dataset and seed the script
evaluates:
  1) PCA-QNN reference;
  2) Zero-QNN implicit baseline;
  3) the preselected MRBI-QNN method;
  4) random-5 multistart QNN control.

The MRBI run uses the same layer construction, profile definitions, solver RNG
offsets, 60-epoch QNN protocol, and QNN seed convention as the development
campaign. All implicit methods use the same QNN seed, so their initial readout
parameters and minibatch order are identical.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
import main_qnn_benchmark as bench
import mrbi

LOCK_PATH=ROOT/"experiments"/"selected_profile_confirmation_lock_v1.json"
PLAN="locked_selected_profile_confirmation_v1"
DEFAULT_OUT=ROOT/"outputs"/PLAN
EXPECTED_LOCK_SCHEMA="mrbi_selected_profile_confirmation_lock_v1"
EXPECTED_RAW_SHA="e1a3917be493485156e87e79c921d2a91a54cf715aa55443516059fc3d370cd6"
EXPECTED_SELECTED_SHA="4cb9ad804076dea25820b73c1640a81e85c7d3b4ad77d16341555269da707908"


def atomic_write(path:Path,text:str):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(text,encoding="utf-8")
    tmp.replace(path)


def git_commit():
    return subprocess.run(
        ["git","rev-parse","HEAD"],cwd=ROOT,
        capture_output=True,text=True,check=True,
    ).stdout.strip()


def load_lock():
    lock=json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("lock_schema")!=EXPECTED_LOCK_SCHEMA:
        raise SystemExit("Selected-profile lock schema drift")
    ev=lock.get("development_evidence",{})
    if ev.get("main_raw_sha256")!=EXPECTED_RAW_SHA:
        raise SystemExit("Development raw SHA drift in lock")
    if ev.get("selected_profiles_sha256")!=EXPECTED_SELECTED_SHA:
        raise SystemExit("Development selected-profile SHA drift in lock")
    datasets=tuple(ev.get("datasets",()))
    if len(datasets)!=9 or len(set(datasets))!=9:
        raise SystemExit("Lock must contain exactly nine unique datasets")
    rows=lock.get("selected_profiles",[])
    if len(rows)!=9 or {r["dataset"] for r in rows}!=set(datasets):
        raise SystemExit("Selected profile map is incomplete")
    seeds=tuple(lock["confirmation_plan"]["seeds"])
    if seeds!=tuple(range(40,50)):
        raise SystemExit("Confirmation seed lock drift")
    return lock


def make_cfg(seed:int,lock):
    plan=lock["confirmation_plan"]
    cfg=bench.ExperimentConfig(
        seed=seed,
        test_size=float(plan["test_size"]),
        n_qubits=int(plan["n_qubits"]),
        pca_dim=int(plan["n_qubits"]),
        latent_dim=int(plan["latent_dim"]),
        max_samples_per_class=int(plan["max_samples_per_class"]),
        spectral_radius=float(plan["spectral_radius"]),
        input_scale=float(plan["input_scale"]),
        bias_scale=0.10,
        hard_layer=True,
        run_qnn=True,
        qnn_epochs=int(plan["qnn_epochs"]),
        qnn_lr=float(plan["qnn_learning_rate"]),
        qnn_batch_size=16,
        qnn_layers=int(plan["qnn_layers"]),
    )
    if (
        cfg.n_qubits!=4 or cfg.latent_dim!=16 or
        cfg.spectral_radius!=2.0 or cfg.input_scale!=1.1 or
        cfg.max_samples_per_class!=80 or cfg.qnn_epochs!=60 or
        cfg.qnn_lr!=0.01
    ):
        raise SystemExit("Confirmation configuration drift")
    return cfg


def selected_cfg(dataset:str,cfg,lock):
    row=next(r for r in lock["selected_profiles"] if r["dataset"]==dataset)
    profiles={p.name:p for p in bench.get_mrbi_profiles("hard_quick")}
    hybrids={h.name:h for h in bench.get_hybrid_profiles("hard_quick")}
    if row["profile"] not in profiles:
        raise SystemExit(f"Unknown locked MRBI profile {row['profile']}")
    cfg_sel=bench.apply_mrbi_profile(cfg,profiles[row["profile"]])
    if row["kind"]=="hybrid":
        if row["hybrid"] not in hybrids:
            raise SystemExit(f"Unknown locked hybrid {row['hybrid']}")
        cfg_sel=bench.apply_hybrid_profile(cfg_sel,hybrids[row["hybrid"]])
    elif row["kind"]!="forced":
        raise SystemExit(f"Unknown locked method kind {row['kind']}")
    return row,cfg_sel


def solve_selected(Xtr,Xte,layer,cfg,row):
    """Reproduce the exact representation path used by development campaign."""
    if row["kind"]=="forced":
        Ztr,st_tr=bench.solve_forced_mrbi_features(
            Xtr,layer,cfg,cfg.seed+2000,
        )
        Zte,st_te=bench.solve_forced_mrbi_features(
            Xte,layer,cfg,cfg.seed+3000,
        )
    else:
        Ztr,st_tr=bench.solve_hybrid_features(
            Xtr,layer,cfg,cfg.seed+4000,
        )
        Zte,st_te=bench.solve_hybrid_features(
            Xte,layer,cfg,cfg.seed+5000,
        )
    return Ztr,Zte,st_tr,st_te


def choose_multistart(zero,results):
    candidates=[zero]+list(results)
    successful=[
        r for r in candidates
        if r.success and np.isfinite(r.residual)
        and np.all(np.isfinite(r.z_star))
    ]
    pool=successful if successful else [
        r for r in candidates
        if np.isfinite(r.residual) and np.all(np.isfinite(r.z_star))
    ]
    if not pool:
        return zero
    return min(
        pool,
        key=lambda r:(
            float(r.residual),
            int(r.nfev) if int(r.nfev)>=0 else 10**9,
        ),
    )


def solve_multistart(X,layer,cfg,rng_seed:int,n_starts=5,radius=1.0):
    F,J=mrbi.make_residual_and_jacobian(layer)
    root_cfg=bench.make_root_cfg(cfg)
    rng=np.random.default_rng(rng_seed)
    zero_init=np.zeros(layer.d,dtype=np.float64)
    Z=[]
    rows=[]
    for pos,x in enumerate(np.asarray(X,dtype=np.float64)):
        t0=time.perf_counter()
        zero=mrbi.solve_root(F,J,x,zero_init,cfg=root_cfg)
        others=[]
        for _ in range(n_starts):
            z0=rng.uniform(-radius,radius,size=layer.d)
            others.append(mrbi.solve_root(F,J,x,z0,cfg=root_cfg))
        best=choose_multistart(zero,others)
        Z.append(best.z_star)
        rows.append({
            "position":pos,
            "success":bool(best.success),
            "zero_success":bool(zero.success),
            "rescued":bool((not zero.success) and best.success),
            "residual":float(best.residual),
            "total_nfev":int(
                max(0,zero.nfev)+sum(max(0,r.nfev) for r in others)
            ),
            "runtime_sec":float(time.perf_counter()-t0),
        })
    return np.asarray(Z,dtype=np.float64),{
        "success_rate":float(np.mean([r["success"] for r in rows])),
        "zero_success_rate":float(np.mean([r["zero_success"] for r in rows])),
        "rescues":int(sum(r["rescued"] for r in rows)),
        "mean_residual":float(np.mean([r["residual"] for r in rows])),
        "median_residual":float(np.median([r["residual"] for r in rows])),
        "mean_total_nfev":float(np.mean([r["total_nfev"] for r in rows])),
        "mean_runtime_sec_per_sample":float(np.mean([
            r["runtime_sec"] for r in rows
        ])),
    }


def qnn_metrics(Xtr,ytr,Xte,yte,cfg):
    # Use the same QNN convention as the development campaign.
    return bench.train_qnn(
        Xtr,ytr,Xte,yte,cfg,cfg.seed+777,
    )


def run_dataset(dataset:str,seed:int,dataset_index:int,lock):
    cfg=make_cfg(seed,lock)
    row,cfg_sel=selected_cfg(dataset,cfg,lock)

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

    # Exact corrected-continuation layer seed.
    layer_seed=(
        cfg.seed+1000+cfg.latent_dim+int(100*cfg.spectral_radius)
    )
    layer=bench.make_random_implicit_layer(
        dx=Xtr_p.shape[1],
        d=cfg.latent_dim,
        cfg=cfg,
        seed=layer_seed,
    )

    t0=time.perf_counter()
    Ztr_zero,st_tr_zero=bench.solve_zero_features(Xtr_p,layer,cfg)
    Zte_zero,st_te_zero=bench.solve_zero_features(Xte_p,layer,cfg)
    zero_feature_time=float(time.perf_counter()-t0)

    t0=time.perf_counter()
    Ztr_sel,Zte_sel,st_tr_sel,st_te_sel=solve_selected(
        Xtr_p,Xte_p,layer,cfg_sel,row,
    )
    selected_feature_time=float(time.perf_counter()-t0)

    ms_cfg=lock["confirmation_plan"]["random_multistart"]
    t0=time.perf_counter()
    Ztr_ms,st_tr_ms=solve_multistart(
        Xtr_p,layer,cfg,
        rng_seed=cfg.seed+6000+100*dataset_index,
        n_starts=int(ms_cfg["n_random_starts"]),
        radius=1.0,
    )
    Zte_ms,st_te_ms=solve_multistart(
        Xte_p,layer,cfg,
        rng_seed=cfg.seed+7000+100*dataset_index,
        n_starts=int(ms_cfg["n_random_starts"]),
        radius=1.0,
    )
    multistart_feature_time=float(time.perf_counter()-t0)

    # Match the development campaign: standardize each implicit representation
    # from its training features only.
    Ztr_zero,Zte_zero=bench.standardize_pair(Ztr_zero,Zte_zero)
    Ztr_sel,Zte_sel=bench.standardize_pair(Ztr_sel,Zte_sel)
    Ztr_ms,Zte_ms=bench.standardize_pair(Ztr_ms,Zte_ms)

    results={
        "pca":qnn_metrics(Xtr_p,ytr,Xte_p,yte,cfg),
        "zero":qnn_metrics(Ztr_zero,ytr,Zte_zero,yte,cfg),
        "selected_mrbi":qnn_metrics(Ztr_sel,ytr,Zte_sel,yte,cfg),
        "multistart5":qnn_metrics(Ztr_ms,ytr,Zte_ms,yte,cfg),
    }
    p=float(results["pca"]["balanced_accuracy"])
    z=float(results["zero"]["balanced_accuracy"])
    m=float(results["selected_mrbi"]["balanced_accuracy"])
    s=float(results["multistart5"]["balanced_accuracy"])

    return {
        "dataset":dataset,
        "seed":seed,
        "selected_method":row["method"],
        "selected_kind":row["kind"],
        "selected_profile":row["profile"],
        "selected_hybrid":row["hybrid"],
        "development_mean_ba":row["development_mean_ba"],
        "layer_seed":layer_seed,
        "qnn_seed":cfg.seed+777,
        "n_train":int(len(ytr)),
        "n_test":int(len(yte)),
        "results":results,
        "paired":{
            "delta_selected_zero":m-z,
            "delta_selected_multistart5":m-s,
            "delta_selected_pca":m-p,
            "delta_multistart5_zero":s-z,
        },
        "solver":{
            "zero_train":st_tr_zero,
            "zero_test":st_te_zero,
            "selected_train":st_tr_sel,
            "selected_test":st_te_sel,
            "multistart_train":st_tr_ms,
            "multistart_test":st_te_ms,
            "feature_time_zero_sec":zero_feature_time,
            "feature_time_selected_sec":selected_feature_time,
            "feature_time_multistart5_sec":multistart_feature_time,
        },
    }


def summarize_seed(rows):
    out={}
    for key in (
        "delta_selected_zero",
        "delta_selected_multistart5",
        "delta_selected_pca",
        "delta_multistart5_zero",
    ):
        vals=np.asarray([r["paired"][key] for r in rows],dtype=float)
        out[key]={
            "mean":float(vals.mean()),
            "median":float(np.median(vals)),
            "positive":int(np.sum(vals>1e-12)),
            "neutral":int(np.sum(np.abs(vals)<=1e-12)),
            "negative":int(np.sum(vals< -1e-12)),
        }
    for method in ("pca","zero","selected_mrbi","multistart5"):
        vals=np.asarray([
            r["results"][method]["balanced_accuracy"] for r in rows
        ],dtype=float)
        out[f"{method}_balanced_accuracy_mean"]=float(vals.mean())
    return out


def dry_run(seed:int,lock):
    plan=lock["confirmation_plan"]
    if seed not in plan["seeds"]:
        raise SystemExit("Seed outside the fixed confirmation set")
    print(
        "LOCKED_SELECTED_PROFILE_CONFIRMATION_DRY_RUN_OK "
        f"seed={seed} datasets=9 rho={plan['spectral_radius']} "
        f"qnn_epochs={plan['qnn_epochs']} development_raw_sha="
        f"{lock['development_evidence']['main_raw_sha256'][:12]} "
        "profiles=fixed random5=fixed no_confirmation_data_loaded",
        flush=True,
    )
    for row in lock["selected_profiles"]:
        print(
            f"LOCKED_PROFILE {row['dataset']} -> {row['method']}",
            flush=True,
        )


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run",action="store_true")
    mode.add_argument("--run",action="store_true")
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--out-dir",type=Path,default=None)
    args=parser.parse_args()

    lock=load_lock()
    if args.seed not in lock["confirmation_plan"]["seeds"]:
        raise SystemExit("Seed is not in the fixed confirmation set 40-49")
    if args.dry_run:
        dry_run(args.seed,lock)
        return
    if not bench.HAS_QNN:
        raise SystemExit(
            f"QNN dependency unavailable: {bench.QNN_IMPORT_ERROR}"
        )

    commit=git_commit()
    out_dir=(
        args.out_dir.resolve() if args.out_dir is not None
        else (DEFAULT_OUT/f"seed{args.seed}").resolve()
    )
    out_dir.mkdir(parents=True,exist_ok=True)

    manifest={
        "plan":PLAN,
        "git_commit":commit,
        "lock_file":str(LOCK_PATH.relative_to(ROOT)),
        "development_evidence":lock["development_evidence"],
        "confirmation_plan":lock["confirmation_plan"],
        "seed":args.seed,
        "selection_map":{
            r["dataset"]:r["method"] for r in lock["selected_profiles"]
        },
    }
    manifest_path=out_dir/"manifest.json"
    raw_manifest=json.dumps(manifest,indent=2,sort_keys=True)+"\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8")!=raw_manifest:
        raise SystemExit("Existing manifest differs; use a new output directory")
    atomic_write(manifest_path,raw_manifest)

    checkpoint=out_dir/"checkpoint.json"
    rows=[]
    if checkpoint.exists():
        old=json.loads(checkpoint.read_text(encoding="utf-8"))
        if old.get("git_commit")!=commit or old.get("seed")!=args.seed:
            raise SystemExit("Checkpoint provenance mismatch")
        rows=old.get("dataset_results",[])
    completed={r["dataset"] for r in rows}

    datasets=lock["development_evidence"]["datasets"]
    for index,dataset in enumerate(datasets):
        if dataset in completed:
            print(f"SKIP completed dataset={dataset}",flush=True)
            continue
        selected=next(
            r["method"] for r in lock["selected_profiles"]
            if r["dataset"]==dataset
        )
        print(
            f"CONFIRM_START seed={args.seed} dataset={dataset} "
            f"method={selected}",
            flush=True,
        )
        result=run_dataset(dataset,args.seed,index,lock)
        rows.append(result)
        atomic_write(
            checkpoint,
            json.dumps({
                "git_commit":commit,
                "seed":args.seed,
                "dataset_results":rows,
            },indent=2,allow_nan=False)+"\n",
        )
        p=result["paired"]
        print(
            f"CONFIRM_DONE dataset={dataset} "
            f"zero={result['results']['zero']['balanced_accuracy']:.4f} "
            f"selected={result['results']['selected_mrbi']['balanced_accuracy']:.4f} "
            f"ms5={result['results']['multistart5']['balanced_accuracy']:.4f} "
            f"pca={result['results']['pca']['balanced_accuracy']:.4f} "
            f"d_sz={p['delta_selected_zero']:+.4f} "
            f"d_sms={p['delta_selected_multistart5']:+.4f}",
            flush=True,
        )

    if len(rows)!=9:
        raise RuntimeError("Incomplete nine-dataset confirmation seed")
    final={
        **manifest,
        "dataset_results":rows,
        "seed_summary":summarize_seed(rows),
    }
    path=out_dir/"locked_result.json"
    raw=json.dumps(final,indent=2,allow_nan=False)+"\n"
    if path.exists() and path.read_text(encoding="utf-8")!=raw:
        raise SystemExit("Existing final result differs; use a new output directory")
    atomic_write(path,raw)
    print(f"LOCKED_SELECTED_PROFILE_CONFIRMATION_OK {path}",flush=True)


if __name__=="__main__":
    main()
