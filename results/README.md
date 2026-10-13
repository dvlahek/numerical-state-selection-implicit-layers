# Aggregate results

This directory contains compact aggregate outputs used to verify the reported numerical results. It is intentionally not a complete archive of per-seed runs, feature matrices, masks, predictions, or other intermediate files.

The main groups are:

- locked confirmation summaries
- spectral-radius and returned-state summaries
- matched Hybrid-Random and Hybrid-LS controls
- classical and QNN readout summaries
- both-strict and common-mask selector analyses
- common-mask state-retention decomposition
- local feature-geometry and fixed-point-attractivity summaries

The compact state-retention table is stored in
`final/common_mask_state_decomposition_summary.csv`.

Large generated outputs are recreated by the scripts in `scripts/` and written under `outputs/`, which is excluded from version control.
