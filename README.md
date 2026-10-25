# Telling Cause from Correlation on Corr2Cause

A neuro-symbolic pipeline that decides whether a causal claim follows from a set of correlation and independence
facts written in text. It uses the [Corr2Cause](https://huggingface.co/datasets/causal-nlp/corr2cause) dataset
(ICLR 2024).

**How it works:** a regex parser reads each premise, and pgmpy's PC algorithm turns it into a causal graph.
Classical models (Logistic Regression, Random Forest, Gradient Boosting) then predict which claims are valid. The
predicted cause arrows are built into a DAG and a pgmpy Bayesian network, which answers intervention ("what if")
questions.

## Results (test set)

| What we measured | Result |
|---|---|
| Symbolic rule (parser + PC, no learning): share of labels matched | **98.5%** (F1 0.952) |
| Random Forest, chosen on validation: precision / recall / F1 | 0.927 / 0.989 / **0.957** |
| 95% bootstrap confidence interval for F1 (resampling premises) | 0.921 – 0.982 |
| Validation F1: TF-IDF only vs TF-IDF + structured features | 0.47 vs 0.88 |
| Edge precision of the predicted DAGs (against the PC graph) | 0.89 (1.00) |
| Intervention queries correct on the 5 fully known test graphs | **92 / 96**, including 28 / 32 real effects |

For comparison, an empty graph or a reversed graph gets **0 of the 32** real effects right.

## Pipeline

| Step | Notebook | Owner | What it does |
|---|---|---|---|
| 1 | `01_preprocessing_eda` | Chaitanya Sai Meka ([@ChaitanyaSai-Meka](https://github.com/ChaitanyaSai-Meka)) | Cleaning, leakage removal, subsampling train, EDA |
| 2 | `02_parser_pc` | Rakshita ([@Rakshita-0023](https://github.com/Rakshita-0023)) | Regex parser, schema checks, PC algorithm, equivalence class, `mec_support` |
| 3 | `03_feature_engineering` | Dhanvin Vadlamudi ([@Dhanvin1520](https://github.com/Dhanvin1520)) | TF-IDF, 15 structured features, chi-squared / mutual information, ablation |
| 4 | `04_models_evaluation` | Kasula Lalithendra ([@Lalith0024](https://github.com/Lalith0024)) | GridSearchCV with GroupKFold, model selection, confusion matrix, bootstrap CI, error analysis |
| 5 | `05_dag_bayesian_network` | Jagruthi Pulumati ([@Jag2007](https://github.com/Jag2007)) | DAG building, edge precision, Bayesian network, intervention queries |

## Getting started

Requires Python 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the notebooks in `notebooks/` in order (01 → 05). Notebook 01 downloads the data from Hugging Face. Each
notebook writes its output as CSV files to `data/processed/` for the next notebook, and saves its figures to
`report/figures/`. A full run takes about 4 minutes.

> **Note:** the processed CSVs (about 140 MB) and the raw data are not stored in git. Running the notebooks in order
> rebuilds them.

Run the unit tests:

```bash
python -m pytest
```

## Repository layout

```
notebooks/        01 → 05, one per team member
src/c2c/          shared code used by the notebooks
  parsing.py      regex parser and schema checks        (notebook 02)
  graphs.py       PC, equivalence class, claim rules    (notebook 02)
  features.py     structured features                   (notebook 03)
  bayes.py        DAG building, intervention effects    (notebook 05)
tests/            pytest unit tests for every module
report/           IEEE report (main.tex, main.pdf), figures, phase 1 slides
data/             raw/ and processed/ CSVs, rebuilt by the notebooks (not in git)
```

## Evaluation protocol

* **Train:** fits everything (TF-IDF, scaler, models). It is subsampled to 400 whole premises (seed 42).
* **Validation:** used to choose the final model.
* **Test:** used once, only to report the final results.

Premises shared between splits are removed, so there is no data leakage. Cross-validation (GroupKFold) and the
bootstrap both group rows by premise.

## Report

The full write-up is in [`report/main.pdf`](report/main.pdf). To rebuild it from the LaTeX source, run
`tectonic main.tex` inside `report/`.
