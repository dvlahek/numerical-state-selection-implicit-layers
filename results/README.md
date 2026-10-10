# Reported results

This directory contains compact aggregate outputs used in the manuscript and supplementary information.

The files are reported summaries, not a complete dump of intermediate caches.

## Locked confirmation

- `final/confirmation_summary.json`
- `final/confirmation_dataset_summary.csv`

Primary Selected MRBI-QNN versus Zero-QNN confirmation.

## Numerical state selection

- `final/all_path_rho_selector_summary.json`
- `final/rho_multistart_prediction_flip_summary.json`
- `final/branch_state_summary.json`

Spectral-radius sweep, strict-root multiplicity, returned-state disagreement, and downstream prediction changes.

## Selector controls

- `final/hybrid_random_control_summary.json`
- `final/hybrid_random_control_dataset_summary.csv`
- `final/hybrid_least_squares_control_summary.json`
- `final/hybrid_least_squares_control_dataset_summary.csv`

Matched Hybrid-Random and Hybrid-LS controls.

## Downstream readouts

- `final/all_path_classical_readouts_summary.json`
- `final/qnn_multiseed_all_paths_summary.json`
- `final/qnn_seed_sensitivity_summary.json`

Classical and QNN downstream controls.

## Both-strict and common-mask analyses

- `final/selection_quality_both_strict_overall.csv`
- `final/selection_quality_qnn_both_strict_summary.json`
- `final/selection_quality_qnn_both_strict_overall.csv`
- `final/selection_quality_qnn_both_strict_dataset_summary.csv`
- `final/selector_confound_compact_summary.csv`
- `final/trigger_conditioned_multistart_summary.csv`
- `final/selector_common_mask_summary.csv`
- `final/selector_common_mask_state_overlap.csv`
- `final/selector_common_mask_bootstrap_summary.csv`

These are post-confirmation descriptive analyses. Task-equal summaries first average seeds within each task and then weight tasks equally.

## Geometry and local fixed-point attractivity

- `final/local_feature_geometry_summary.json`
- `final/local_feature_geometry_strict_k3.csv`
- `final/equilibrium_stability_summary.csv`

The attractivity diagnostic uses the spectral radius of the Jacobian of the natural tanh fixed-point map. It does not measure `hybr` solver convergence.
