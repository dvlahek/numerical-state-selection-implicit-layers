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
- local feature geometry
- local fixed-point attractivity

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
- task-equal aggregation
- local fixed-point attractivity

Local fixed-point attractivity refers to the natural tanh fixed-point map and is not a convergence certificate for the final `hybr` root solver.
