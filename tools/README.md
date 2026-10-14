# Tools

`tools/` contains code that supports the research workflow but is not part of
the paper training entry point.

- `data/`: dataset preparation and the MMLSv2 → Mars-LS representation mapping.
- `evaluation/`: frozen-checkpoint validation/test evaluation. These scripts
  must only be used after model and checkpoint selection are frozen.
- `analysis/`: result tables, historical log parsing, visualization and legacy
  analysis notes.
- `validation/`: data-only smoke checks.
- `knowledge/`: train-only prototype artifact construction used by the SKG
  training pipeline; it does not optimize a segmentation model.
- `submission/`: historical competition inference and zip generation. These
  scripts are intentionally excluded from the paper experiment protocol.

The paper training entry points remain in the repository root and `scripts/`.
