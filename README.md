# Numerical State Selection Exposes Representation Ambiguity in Nonunique Implicit Neural Layers

Minimal reproducibility repository for the manuscript **Numerical State Selection Exposes Representation Ambiguity in Nonunique Implicit Neural Layers**.

The study isolates numerical state selection in a fixed nonunique implicit layer. The implicit operator and final root solver are held fixed while the numerical rule that exposes one solver-accessible state is varied. The returned state is then used as a downstream feature.

The five numerical paths are:

- **Zero** — zero initialization followed by the common final root solver.
- **Selected MRBI** — Multiscale Residual-Based Initialization (MRBI) constructs an input-specific candidate before the same final root solver.
- **Hybrid-LS** — a matched hybrid policy using a bounded least-squares candidate.
- **Hybrid-Random** — the matched hybrid policy using a random candidate.
- **Multistart-5** — zero plus five random starts, with the strict solution of smallest residual selected.

The main experimental separation is

```text
strict recovery != numerical state selection != downstream feature utility
```

Post-confirmation analyses additionally separate state retention and local fixed-point attractivity from downstream utility.

## Primary downstream test

The pre-specified primary readout is a simulated four-qubit quantum neural network (QNN). For implicit features, the returned 16-dimensional state is standardized and mapped through a trainable 16-to-4 classical interface to the same four-angle, four-qubit circuit. Five fixed classical readouts are used as post-confirmation controls for readout dependence.

No quantum-advantage or quantum-hardware claim is made.

## Locked protocol

- development seeds: `0..4`
- confirmation seeds: `40..49`
- nine binary tasks derived from scikit-learn datasets
- compact input: four PCA coordinates followed by tanh bounding
- implicit state dimension: 16
- confirmation spectral radius: `rho(W)=2.0`
- spectral-radius sweep: `{1.2, 1.6, 2.0, 2.25}`
- strict residual threshold: `1e-8`
- state-agreement threshold: relative distance `<=1e-6`
- final nonlinear solver: `scipy.optimize.root(..., method="hybr")`

The task-specific MRBI profile map was selected on development seeds and locked before the confirmation run. It is stored in `experiments/selected_profile_confirmation_lock_v1.json`.

## Repository layout

```text
experiments/   core implicit-layer, MRBI, and locked-profile code
scripts/       reproduction and post-confirmation diagnostic scripts
results/       compact aggregate results reported in the manuscript
```

Large feature caches, per-sample masks, and QNN prediction files are regenerated locally and are not tracked.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-qnn.txt
```

The QNN experiments use PennyLane analytic simulation.

## Reproduction map

### 1. Locked primary confirmation

```bash
for seed in {40..49}; do
    python -u scripts/run_locked_selected_profile_confirmation.py --run --seed "$seed"
done
python scripts/summarize_locked_selected_profile_confirmation.py
```

This reproduces the pre-specified Selected-MRBI-QNN versus Zero-QNN confirmation.

### 2. Matched selector controls and spectral-radius sweep

```bash
python -u scripts/run_hybrid_random_control.py
python -u scripts/run_least_squares_control.py
python -u scripts/run_all_path_rho_selector_sweep.py
```

This reproduces strict recovery, discovered multiplicity, returned-state disagreement, and the five-path selector comparison.

### 3. QNN and classical readout controls

```bash
python -u scripts/run_qnn_multiseed_all_paths.py --replicates 4
python -u scripts/run_all_path_classical_readouts_rho2.py
```

### 4. Strict-root and common-mask state-selection analyses

```bash
python -u scripts/run_selection_quality_both_strict_audit.py
python -u scripts/run_qnn_both_strict_selection_quality.py
python -u scripts/run_selector_confound_audit.py
python -u scripts/run_common_mask_state_decomposition.py --run
```

The common all-strict mask requires all five paths to return strict states on the same test case and at least one alternative path to differ from Zero. The final command splits that common mask into selector-specific same-state and different-state subsets without regenerating implicit features or retraining QNNs. Its compact manuscript-facing aggregate is tracked in `results/final/common_mask_state_decomposition_summary.csv`.

### 5. Post-confirmation mechanism diagnostics

```bash
python -u scripts/run_local_feature_regularity_raw.py --run
python -u scripts/run_local_feature_regularity_robustness.py --run
python -u scripts/run_equilibrium_stability_audit.py --run
```

The last script measures local fixed-point attractivity under the natural tanh map using

```text
spectral_radius(diag(1-z^2) W) < 1
```

This is not a convergence certificate for the final HYBR root solver.

## Verification

The first reproducibility-complete public computational baseline is commit

`e75af819c471e091fa40ec382901757b93256097`.

The manuscript-complete computational snapshot, including the common-mask state-retention decomposition and its tracked aggregate check, is

`efc45d66136f28c997b5eb5ac52a2cb010dc32c4`.

GitHub Actions workflow run `38453062621` completed successfully for this snapshot. The workflow checked Python source compilation, tracked aggregate-result invariants, the locked confirmation protocol in dry-run mode, and the selector-sweep protocol in dry-run mode.

The aggregate checks can be run locally with

```bash
python scripts/check_final_results.py
python scripts/check_state_decomposition_results.py
```

Detailed source and public-verification provenance is recorded in `PROVENANCE.md`.

## Interpretation

The repository is organized around the manuscript's central result: numerical validity, selected representation, local fixed-point attractivity, and downstream utility are distinct properties. High strict-root recovery alone does not define a useful implicit feature.

## Citation

Citation metadata are provided in `CITATION.cff`. For exact reproducibility of the final manuscript results, use computational snapshot `efc45d66136f28c997b5eb5ac52a2cb010dc32c4`.

## License

MIT. See `LICENSE`.
