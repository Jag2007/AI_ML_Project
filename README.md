# AI_ML_Project · Causal inference on Corr2Cause

We check whether a causal claim follows from a set of correlation facts written in text, using the
[Corr2Cause](https://huggingface.co/datasets/causal-nlp/corr2cause) dataset. A regex parser and pgmpy's PC algorithm turn the
text into causal graphs, classical models predict which claims are valid, and the predicted edges are built into a DAG
and a pgmpy Bayesian network.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run the notebooks in `notebooks/` in order (01 → 05). Notebook 01 downloads the data, and each notebook saves CSV
files in `data/processed/` for the next one and its figures in `report/figures/`. A full run takes about 4 minutes.

Run the tests with:

```bash
python -m pytest
```

## Layout

```
src/c2c/        shared code used by the notebooks
  parsing.py    regex parser and schema checks        (notebook 02)
  graphs.py     PC, equivalence class, claim rules    (notebook 02)
  features.py   structured features                   (notebook 03)
  bayes.py      DAG building, intervention effects    (notebook 05)
tests/          pytest tests for every module
notebooks/      01 → 05, one per team member
report/         IEEE report (main.tex, main.pdf) and figures
data/processed/ CSVs written by the notebooks (not in git)
```

## Team

| Notebook | Owner | Work |
|---|---|---|
| 01_preprocessing_eda | Chaitanya Sai Meka (@ChaitanyaSai-Meka) | cleaning, leakage check, subsample, EDA |
| 02_parser_pc | Rakshita (@Rakshita-0023) | parser, A–F schema, self-loop rule, PC algorithm |
| 03_feature_engineering | Dhanvin Vadlamudi (@Dhanvin1520) | TF-IDF, structured features, selection, importance |
| 04_models_evaluation | Kasula Lalithendra (@Lalith0024) | LogReg / Random Forest / Gradient Boosting, metrics, error analysis |
| 05_dag_bayesian_network | Jagruthi Pulumati (@Jag2007) | DAG rules, edge precision, Bayesian network, interventions |

## Splits

* **train**: fit everything (subsampled to 400 premises with seed 42)
* **validation**: choose the model and threshold
* **test**: final results only

Premises shared between splits are removed, so there is no leakage.

## Results (test)

* Parser + PC with the Corr2Cause generator's claim definitions reproduce **98.5%** of test labels with no learning
  (rule F1 0.952)
* Random Forest (chosen on validation): precision 0.927, recall 0.989, **F1 0.957** (95% CI 0.921–0.982)
* TF-IDF alone reaches F1 0.47 on validation, while TF-IDF + structured features reach 0.88
* Edge precision 0.89 (1.00 against the PC graph)
* Intervention queries on the 5 fully known test graphs: **92 of 96** correct, including 28 of the 32 real effects
  (an empty or reversed graph gets 0 of 32)
