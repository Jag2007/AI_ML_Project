# AI_ML_Project · Causal inference on Corr2Cause

We check whether a causal claim follows from a set of correlation facts written in text, using the
[Corr2Cause](https://huggingface.co/datasets/causal-nlp/corr2cause) dataset. A regex parser and the PC algorithm turn the
text into causal graphs, classical models predict which claims are valid, and the predicted edges are built into a DAG
and a pgmpy Bayesian network.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run the notebooks in `notebooks/` in order (01 → 05). Notebook 01 downloads the data, and each notebook saves CSV
files in `data/processed/` for the next one. Parser tests: `pytest tests`.

## Team

| Notebook | Owner | Work |
|---|---|---|
| 01_preprocessing_eda | Chaitanya Sai Meka (@ChaitanyaSai-Meka) | cleaning, leakage check, subsample, EDA |
| 02_parser_pc | Rakshita (@Rakshita-0023) | parser, A–F schema, self-loop rule, tests, PC algorithm |
| 03_feature_engineering | Dhanvin Vadlamudi (@Dhanvin1520) | TF-IDF, structured features, selection, importance |
| 04_models_evaluation | Kasula Lalithendra (@Lalith0024) | LogReg / Random Forest / Gradient Boosting, metrics, error analysis |
| 05_dag_bayesian_network | Jagruthi Pulumati (@Jag2007) | DAG rules, edge precision, Bayesian network, interventions |

## Splits

* **train**: fit everything (subsampled to 400 premises with seed 42)
* **validation**: choose the model and threshold
* **test**: final results only

Premises shared between splits are removed, so there is no leakage.

## Results (test)

* Random Forest: precision 0.928, recall 0.928, **F1 0.928** (95% CI 0.893–0.957)
* TF-IDF alone reaches F1 0.47 on validation, while TF-IDF + structured features reach 0.79
* Edge precision 0.89 (1.00 against the PC graph), and 7 of 8 intervention queries match the true graph
