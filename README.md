# Numerical State Selection Exposes Representation Ambiguity in Nonunique Implicit Neural Layers

Reproducibility code for the numerical state-selection experiments in the associated study. The implicit operator and final root solver are fixed while the numerical rule used to expose one solver-accessible state is varied.

The repository contains the numerical implementation, locked experimental configuration, reproduction scripts, and compact aggregate results. Manuscript sources, large feature caches, per-sample outputs, and saved prediction arrays are not tracked.

## Numerical paths

- **Zero** — zero initialization followed by the common final root solver
- **Selected MRBI** — Multiscale Residual-Based Initialization followed by the same final root solver
- **Hybrid-LS** — matched least-squares candidate under the same hybrid policy
- **Hybrid-Random** — matched random candidate under the same hybrid policy
- **Multistart-5** — Zero plus five random starts, selecting the strict solution with the smallest residual

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-qnn.txt
```

The QNN experiments use PennyLane analytic simulation.

## Locked protocol

- development seeds: `0..4`
- confirmation seeds: `40..49`
- nine binary tasks derived from scikit-learn datasets
- four PCA coordinates followed by tanh bounding
- implicit state dimension: `16`
- confirmation spectral radius: `2.0`
- spectral-radius sweep: `{1.2, 1.6, 2.0, 2.25}`
- strict residual threshold: `1e-8`
- state-agreement threshold: relative distance `<=1e-6`
- final nonlinear solver: `scipy.optimize.root(..., method="hybr")`

The task-specific MRBI profile map is stored in `experiments/selected_profile_confirmation_lock_v1.json` and is not reselected during confirmation.

## Reproduction

### Primary confirmation

```bash
for seed in {40..49}; do
    python -u scripts/run_locked_selected_profile_confirmation.py --run --seed "$seed"
done
python scripts/summarize_locked_selected_profile_confirmation.py
```

### Selector controls and spectral-radius sweep

```bash
python -u scripts/run_hybrid_random_control.py
python -u scripts/run_least_squares_control.py
python -u scripts/run_all_path_rho_selector_sweep.py
```

### QNN and classical readout controls

```bash
python -u scripts/run_qnn_multiseed_all_paths.py --replicates 4
python -u scripts/run_all_path_classical_readouts_rho2.py
```

### Post-confirmation selector analyses

```bash
python -u scripts/run_selection_quality_both_strict_audit.py
python -u scripts/run_qnn_both_strict_selection_quality.py
python -u scripts/run_selector_confound_audit.py
python -u scripts/run_common_mask_state_decomposition.py --run
python -u scripts/run_local_feature_regularity_raw.py --run
python -u scripts/run_local_feature_regularity_robustness.py --run
python -u scripts/run_equilibrium_stability_audit.py --run
```

These analyses reuse generated feature matrices where applicable. Large intermediate outputs are written under `outputs/` and are intentionally excluded from version control.

## Results

`results/final/` contains compact aggregate summaries used to check the reported numbers. It is not a dump of every run or intermediate file.

The main aggregate checks are:

```bash
python scripts/check_final_results.py
python scripts/check_state_decomposition_results.py
```

GitHub Actions runs the same lightweight checks on each push. The computational files used for the final study are fixed at commit

`efc45d66136f28c997b5eb5ac52a2cb010dc32c4`

and the corresponding reproducibility workflow completed successfully. Later commits only clean public-facing documentation and ignore rules.

## Citation

Citation metadata are provided in `CITATION.cff`.

## License

MIT. See `LICENSE`.
