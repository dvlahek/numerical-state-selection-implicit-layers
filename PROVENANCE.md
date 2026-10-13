# Computational provenance

This repository is the public reproducibility snapshot for the manuscript

**Numerical State Selection Exposes Representation Ambiguity in Nonunique Implicit Neural Layers**

## Source snapshot

The numerical implementation and reported aggregate outputs were exported from the locked internal development snapshot

`947187beea615ddc2784e3fe41a693980e073d5d`

on 10 October 2026.

Before public-facing cleanup, the exported numerical source files and reported aggregate result files were verified byte-for-byte against that source snapshot.

Two terminology-only edits were then applied to public source comments/output text:

- two historical aggregation labels were updated to the manuscript term `task-equal`

These edits do not change numerical logic, parameters, seeds, solver behavior, state selection, training, or reported results.

## Public verification

The first reproducibility-complete public computational snapshot is commit

`e75af819c471e091fa40ec382901757b93256097`.

The final manuscript adds a post-confirmation common-mask state-retention decomposition. Its public script, compact aggregate output, and CI check are included in the manuscript-complete computational snapshot

`efc45d66136f28c997b5eb5ac52a2cb010dc32c4`.

GitHub Actions workflow run `38453062621` completed successfully for this snapshot. The workflow checked Python source compilation, tracked aggregate-result invariants including the state-retention aggregate, the locked confirmation protocol in dry-run mode, and the selector-sweep protocol in dry-run mode.

Documentation edits after this computational snapshot do not modify numerical source code, locked parameters, or tracked scientific results. For exact final-manuscript reproducibility, the manuscript-complete computational snapshot above is the pinned snapshot.

## Locked confirmation

The pre-specified confirmation uses:

- nine binary classification tasks
- confirmation seeds 40--49
- latent dimension 16
- spectral radius 2.0
- four-qubit simulated QNN
- fixed dataset-specific MRBI profile map selected before confirmation
- no result-dependent profile reselection

The primary confirmatory comparison is Selected MRBI-QNN versus Zero-QNN.

## Post-confirmation analyses

The following analyses are descriptive or mechanistic and do not replace the locked primary confirmation:

- spectral-radius selector sweep
- returned-state disagreement
- strict-root multiplicity
- classical-readout controls
- QNN multiseed robustness
- both-strict state-selection analysis
- trigger-conditioned selector analysis
- common all-strict mask analysis
- common-mask state-retention decomposition
- local feature geometry
- local fixed-point attractivity

The state-retention decomposition reuses the established common mask, cached feature matrices, and saved four-seed QNN predictions. It does not regenerate implicit features or retrain QNNs. Its compact manuscript-facing aggregate is stored in `results/final/common_mask_state_decomposition_summary.csv`.

## Terminology

The manuscript and this public repository use:

- Zero
- Selected MRBI
- Hybrid-LS
- Hybrid-Random
- Multistart-5
- strict recovery
- returned-state disagreement
- numerical state selection
- representation ambiguity
- common all-strict mask
- state-retention decomposition
- task-equal aggregation
- local fixed-point attractivity

Local fixed-point attractivity refers to the natural tanh fixed-point map and is not a convergence certificate for the final `hybr` root solver.
