# ovarian-tabpfn

[![DOI](https://zenodo.org/badge/1411385360.svg)](https://doi.org/10.5281/zenodo.23285891)

Code for the study *Discrimination of benign and malignant ovarian tumors from routine laboratory biomarkers using a tabular foundation model*.

The analysis applies the tabular foundation model TabPFN-3 to routine laboratory and demographic variables, and compares it with random forest, XGBoost, logistic regression and a decision tree using repeated stratified cross-validation.

## Repository structure

```
src/ovarian_tabpfn/   data preparation, models, cross-validation, statistics, plots
scripts/              analysis scripts, run in numerical order
results/              patient-level out-of-fold predictions
data/                 place the dataset here (not included)
models/               place the TabPFN-3 checkpoint here (not included)
```

## Installation

Python 3.10 or later is required.

```bash
git clone https://github.com/Amirnaderiy/ovarian_tabpfn.git
cd ovarian_tabpfn
pip install -r requirements.txt
pip install -e .
```

## Data

The study uses a publicly available dataset (Lu et al., 2020), available from Mendeley Data:
https://data.mendeley.com/datasets/th7fztbrv9/11

Download `Supplementary data 1.xlsx` and place it in `data/`.

## Usage

```bash
python scripts/01_cross_validation.py   # repeated 10 x 10 cross-validation
python scripts/02_evaluate.py           # metrics, statistical tests and figures
python scripts/03_shap.py               # SHAP interpretation
python scripts/04_revision_analyses.py  # additional analyses
```

Each script lists its options with `--help`. Script 02 needs only the predictions in `results/`, so the reported metrics can be reproduced without running TabPFN-3.

## Citation

A citation will be added once the article is published.

## License

MIT. See `LICENSE`.
