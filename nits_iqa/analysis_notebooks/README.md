# NITS-IQA analysis notebooks — English publication adaptations

The seven notebooks here are **not byte-identical** to the historical files.
They were adapted for an English-only repository and executed against the
published 405-row tabular CSV. Historical originals are retained locally.

Changes include:

- English comments, output labels and `Distorted` CSV column name.
- Portable relative CSV loading instead of an unpublished local Excel workbook.
- Cleared historical outputs and removed machine-specific absolute paths.
- Repaired the invalid first cell in `01_data_analysis.ipynb`.
- Reordered validation cells in `02_oracle_comparison.ipynb` to define `oof`
  before using it.
- Added the required setup in `03_oracle_comparison.ipynb`, which previously
  relied on variables from another notebook.
- Replaced the repeated binary `balanced_accuracy_score` calculation with an
  equivalent, faster expression for folds containing both classes.

All seven notebooks completed their code cells against the published CSV in a
local validation environment. This is a **tabular re-execution** only: it does
not reconstruct images, MOS collection or deep-model inference. The original
notebook cells and published adaptations have distinct SHA-256 values recorded
in the root publication manifest.
