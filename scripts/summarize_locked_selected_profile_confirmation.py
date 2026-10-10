"""Summarize the selected-profile confirmation on seeds 40-49.

The statistical unit is the dataset. Each of the nine datasets contributes its
mean paired balanced-accuracy difference across ten confirmation seeds. The
primary comparison is selected MRBI-QNN minus Zero-QNN.
"""
from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
PLAN="locked_selected_profile_confirmation_v1"
BASE=ROOT/"outputs"/PLAN
LOCK_PATH=ROOT/"experiments"/"selected_profile_confirmation_lock_v1.json"


def bootstrap_ci(values,seed,n_boot=20000):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    draws=rng.choice(
        values,size=(n_boot,len(values)),replace=True
    ).mean(axis=1)
    return [
        float(np.quantile(draws,0.025)),
        float(np.quantile(draws,0.975)),
    ]


def exact_signed_rank_greater(values,tol=1e-12):
    """Exact one-sided signed-rank sign-permutation test.

    Values with |delta| <= tol are treated as numerical zero.  This makes the
    final statistic independent of SciPy's version-specific handling of tiny
    floating-point near-zeros in wilcoxon().
    """
    values=np.asarray(values,dtype=float)
    values=values[np.abs(values)>tol]
    if len(values)==0:
        return {
            "statistic":0.0,
            "pvalue":1.0,
            "alternative":"greater",
            "zero_tolerance":tol,
            "method":"exact signed-rank sign permutation",
        }
    ranks=rankdata(np.abs(values),method="average")
    statistic=float(np.sum(ranks[values>0]))
    null_statistics=np.fromiter(
        (
            sum(rank for rank,include in zip(ranks,bits) if include)
            for bits in product((0,1),repeat=len(ranks))
        ),
        dtype=float,
    )
    pvalue=float(np.mean(null_statistics>=statistic-1e-15))
    return {
        "statistic":statistic,
        "pvalue":pvalue,
        "alternative":"greater",
        "zero_tolerance":tol,
        "method":"exact signed-rank sign permutation",
    }

def selected_success_rate(result):
    stats=result["solver"]["selected_test"]
    if result["selected_kind"]=="forced":
        return float(stats["forced_success_rate"])
    return float(stats["hybrid_success_rate"])


def main():
    lock=json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    datasets=tuple(lock["development_evidence"]["datasets"])
    seeds=tuple(lock["confirmation_plan"]["seeds"])
    if datasets!=(
        "breast_cancer","wine_binary","wine_0_vs_2","wine_1_vs_2",
        "digits_1_vs_7","digits_2_vs_7","digits_3_vs_8",
        "digits_4_vs_9","digits_5_vs_6",
    ):
        raise SystemExit("Locked dataset order drift")
    if seeds!=tuple(range(40,50)):
        raise SystemExit("Locked confirmation seed drift")

    records=[]
    commits=set()
    for seed in seeds:
        path=BASE/f"seed{seed}"/"locked_result.json"
        if not path.exists():
            raise SystemExit(f"Missing confirmation result: {path}")
        payload=json.loads(path.read_text(encoding="utf-8"))
        if payload.get("plan")!=PLAN or payload.get("seed")!=seed:
            raise SystemExit(f"Result provenance mismatch: {path}")
        commits.add(payload["git_commit"])
        rows=payload.get("dataset_results",[])
        if tuple(r["dataset"] for r in rows)!=datasets:
            raise SystemExit(f"Incomplete/order-drifted dataset set: {path}")

        for result in rows:
            dataset=result["dataset"]
            expected=next(
                r["method"] for r in lock["selected_profiles"]
                if r["dataset"]==dataset
            )
            if result["selected_method"]!=expected:
                raise SystemExit(
                    f"Selected method drift for {dataset}: "
                    f"{result['selected_method']} != {expected}"
                )
            rec={
                "dataset":dataset,
                "seed":seed,
                "selected_method":expected,
                "development_mean_ba":float(
                    result["development_mean_ba"]
                ),
                "pca":float(
                    result["results"]["pca"]["balanced_accuracy"]
                ),
                "zero":float(
                    result["results"]["zero"]["balanced_accuracy"]
                ),
                "selected_mrbi":float(
                    result["results"]["selected_mrbi"]["balanced_accuracy"]
                ),
                "multistart5":float(
                    result["results"]["multistart5"]["balanced_accuracy"]
                ),
                "delta_selected_zero":float(
                    result["paired"]["delta_selected_zero"]
                ),
                "delta_selected_multistart5":float(
                    result["paired"]["delta_selected_multistart5"]
                ),
                "delta_selected_pca":float(
                    result["paired"]["delta_selected_pca"]
                ),
                "delta_multistart5_zero":float(
                    result["paired"]["delta_multistart5_zero"]
                ),
                "zero_success":float(
                    result["solver"]["zero_test"]["zero_success_rate"]
                ),
                "selected_success":selected_success_rate(result),
                "multistart_success":float(
                    result["solver"]["multistart_test"]["success_rate"]
                ),
                "zero_feature_time":float(
                    result["solver"]["feature_time_zero_sec"]
                ),
                "selected_feature_time":float(
                    result["solver"]["feature_time_selected_sec"]
                ),
                "multistart_feature_time":float(
                    result["solver"]["feature_time_multistart5_sec"]
                ),
            }
            records.append(rec)

    dataset_rows=[]
    for dataset in datasets:
        subset=[r for r in records if r["dataset"]==dataset]
        if len(subset)!=10:
            raise SystemExit(f"{dataset}: expected 10 confirmation seeds")
        methods={r["selected_method"] for r in subset}
        if len(methods)!=1:
            raise SystemExit(f"{dataset}: selected method changed across seeds")
        row={
            "dataset":dataset,
            "selected_method":next(iter(methods)),
            "n_confirmation_seeds":10,
            "development_selected_ba":float(subset[0]["development_mean_ba"]),
        }
        for key in (
            "pca","zero","selected_mrbi","multistart5",
            "delta_selected_zero","delta_selected_multistart5",
            "delta_selected_pca","delta_multistart5_zero",
            "zero_success","selected_success","multistart_success",
            "zero_feature_time","selected_feature_time","multistart_feature_time",
        ):
            row[key]=float(np.mean([r[key] for r in subset]))
        row["development_to_confirmation_selected_ba_shift"]=(
            row["selected_mrbi"]-row["development_selected_ba"]
        )
        row["positive_seed_deltas_selected_zero"]=int(sum(
            r["delta_selected_zero"]>1e-12 for r in subset
        ))
        row["neutral_seed_deltas_selected_zero"]=int(sum(
            abs(r["delta_selected_zero"])<=1e-12 for r in subset
        ))
        row["negative_seed_deltas_selected_zero"]=int(sum(
            r["delta_selected_zero"]< -1e-12 for r in subset
        ))
        dataset_rows.append(row)

    dz=np.asarray([
        r["delta_selected_zero"] for r in dataset_rows
    ],dtype=float)
    dm=np.asarray([
        r["delta_selected_multistart5"] for r in dataset_rows
    ],dtype=float)
    dp=np.asarray([
        r["delta_selected_pca"] for r in dataset_rows
    ],dtype=float)
    dmsz=np.asarray([
        r["delta_multistart5_zero"] for r in dataset_rows
    ],dtype=float)
    solver_gain=np.asarray([
        r["selected_success"]-r["zero_success"]
        for r in dataset_rows
    ],dtype=float)

    overall={
        "pca_ba_mean":float(np.mean([r["pca"] for r in dataset_rows])),
        "zero_ba_mean":float(np.mean([r["zero"] for r in dataset_rows])),
        "selected_mrbi_ba_mean":float(np.mean([
            r["selected_mrbi"] for r in dataset_rows
        ])),
        "multistart5_ba_mean":float(np.mean([
            r["multistart5"] for r in dataset_rows
        ])),
        "delta_selected_zero_mean":float(dz.mean()),
        "delta_selected_zero_median":float(np.median(dz)),
        "delta_selected_zero_bootstrap95_dataset_ci":bootstrap_ci(
            dz,seed=20261040,
        ),
        "delta_selected_zero_positive_datasets":int(np.sum(dz>1e-12)),
        "delta_selected_zero_neutral_datasets":int(
            np.sum(np.abs(dz)<=1e-12)
        ),
        "delta_selected_zero_negative_datasets":int(np.sum(dz< -1e-12)),
        "wilcoxon_selected_gt_zero":exact_signed_rank_greater(dz),
        "delta_selected_multistart5_mean":float(dm.mean()),
        "delta_selected_multistart5_median":float(np.median(dm)),
        "delta_selected_multistart5_bootstrap95_dataset_ci":bootstrap_ci(
            dm,seed=20261041,
        ),
        "wilcoxon_selected_gt_multistart5":exact_signed_rank_greater(dm),
        "delta_selected_pca_mean":float(dp.mean()),
        "delta_multistart5_zero_mean":float(dmsz.mean()),
        "selected_solver_success_gain_vs_zero_mean":float(
            solver_gain.mean()
        ),
        "zero_success_rate_mean":float(np.mean([
            r["zero_success"] for r in dataset_rows
        ])),
        "selected_success_rate_mean":float(np.mean([
            r["selected_success"] for r in dataset_rows
        ])),
        "multistart5_success_rate_mean":float(np.mean([
            r["multistart_success"] for r in dataset_rows
        ])),
        "zero_feature_time_sec_mean":float(np.mean([
            r["zero_feature_time"] for r in dataset_rows
        ])),
        "selected_feature_time_sec_mean":float(np.mean([
            r["selected_feature_time"] for r in dataset_rows
        ])),
        "multistart5_feature_time_sec_mean":float(np.mean([
            r["multistart_feature_time"] for r in dataset_rows
        ])),
        "development_selected_ba_mean":float(np.mean([
            r["development_selected_ba"] for r in dataset_rows
        ])),
        "confirmation_minus_development_selected_ba_mean":float(np.mean([
            r["development_to_confirmation_selected_ba_shift"]
            for r in dataset_rows
        ])),
    }

    summary={
        "plan":PLAN,
        "development_raw_sha256":lock["development_evidence"]["main_raw_sha256"],
        "development_selected_profiles_sha256":lock[
            "development_evidence"
        ]["selected_profiles_sha256"],
        "development_seeds":lock["development_evidence"]["seeds"],
        "confirmation_seeds":list(seeds),
        "n_datasets":len(datasets),
        "n_dataset_seed_pairs":len(records),
        "commits_present":sorted(commits),
        "primary_statistical_unit":"dataset mean across ten confirmation seeds",
        "primary_comparison":"selected MRBI-QNN minus Zero-QNN",
        "dataset_level":dataset_rows,
        "overall":overall,
        "interpretation_notes":[
            "The dataset-specific methods were selected on development seeds 0-4 before confirmation seeds 40-49 were run.",
            "Confirmation results do not change the selected method, dataset set, seed set, or thresholds.",
            "The primary analysis uses nine dataset-level paired differences, each averaged over ten confirmation seeds.",
            "Random-5 multistart and PCA-QNN are secondary references. The primary comparison is selected MRBI-QNN versus Zero-QNN.",
            "The confirmation uses new splits of the same nine benchmark tasks; it is not external-dataset validation.",
        ],
    }

    BASE.mkdir(parents=True,exist_ok=True)
    (BASE/"locked_summary.json").write_text(
        json.dumps(summary,indent=2,allow_nan=False)+"\n",
        encoding="utf-8",
    )
    with (BASE/"locked_dataset_summary.csv").open(
        "w",newline="",encoding="utf-8"
    ) as handle:
        writer=csv.DictWriter(
            handle,fieldnames=list(dataset_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(dataset_rows)

    print(
        "LOCKED_SELECTED_PROFILE_SUMMARY_OK",
        BASE/"locked_summary.json",
        flush=True,
    )
    print(json.dumps(overall,indent=2),flush=True)


if __name__=="__main__":
    main()
