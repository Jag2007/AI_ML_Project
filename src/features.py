"""TF-IDF features from the text and structured features from the parsed facts."""

import math

import networkx as nx
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.parser import HYPOTHESIS_TYPES, parse_hypothesis, parse_premise
from src.pc import markov_equivalence_class, relation_support, run_pc

STRUCTURED_COLUMNS = [
    "parsed_num_variables",
    "n_correlations",
    "n_marginal_independencies",
    "n_conditional_independencies",
    *[f"rel_{r}" for r in HYPOTHESIS_TYPES],
    "xy_correlated",
    "xy_marginally_independent",
    "xy_any_independence",
    "xy_adjacent",
    "edge_x_to_y",
    "edge_y_to_x",
    "edge_undirected",
    "directed_path_x_to_y",
    "n_common_children",
    "n_common_parents",
    "n_common_neighbours",
    "log_mec_size",
    "mec_support",
]

# Keep one-letter tokens, because the variable names are A, B, C...
TOKEN_PATTERN = r"(?u)\b\w+\b"


def _row_features(facts, cpdag, mec, hyp):
    x, y = hyp.x, hyp.y
    directed = nx.DiGraph(list(cpdag.directed))
    directed.add_nodes_from(facts.variables)
    neighbours = {v: {u for u in facts.variables if cpdag.adjacent(u, v)} for v in facts.variables}
    others = [z for z in facts.variables if z not in (x, y)]
    row = {
        "parsed_num_variables": facts.num_variables,
        "n_correlations": len(facts.correlations),
        "n_marginal_independencies": sum(1 for *_, c in facts.independencies if not c),
        "n_conditional_independencies": sum(1 for *_, c in facts.independencies if c),
        "xy_correlated": int(frozenset((x, y)) in facts.correlations),
        "xy_marginally_independent": int(facts.is_independent(x, y)),
        "xy_any_independence": int(any({a, b} == {x, y} for a, b, _ in facts.independencies)),
        "xy_adjacent": int(cpdag.adjacent(x, y)),
        "edge_x_to_y": int((x, y) in cpdag.directed),
        "edge_y_to_x": int((y, x) in cpdag.directed),
        "edge_undirected": int(frozenset((x, y)) in cpdag.undirected),
        "directed_path_x_to_y": int(nx.has_path(directed, x, y)),
        "n_common_children": sum((x, z) in cpdag.directed and (y, z) in cpdag.directed for z in others),
        "n_common_parents": sum((z, x) in cpdag.directed and (z, y) in cpdag.directed for z in others),
        "n_common_neighbours": len(neighbours[x] & neighbours[y]),
        "log_mec_size": math.log(len(mec)),
        "mec_support": relation_support(mec, hyp.relation, x, y),
    }
    for r in HYPOTHESIS_TYPES:
        row[f"rel_{r}"] = int(hyp.relation == r)
    return row


def add_structured_features(df):
    """Adds the structured columns. Pure rules, no fitting and no labels, so it's safe on every split."""
    cache, rows = {}, []
    for pid, premise, hypothesis in zip(df["premise_id"], df["premise"], df["hypothesis"]):
        if pid not in cache:
            facts = parse_premise(premise)
            cpdag = run_pc(facts)
            cache[pid] = (facts, cpdag, markov_equivalence_class(cpdag))
        facts, cpdag, mec = cache[pid]
        rows.append(_row_features(facts, cpdag, mec, parse_hypothesis(hypothesis, facts.variables)))
    return pd.concat([df, pd.DataFrame(rows, index=df.index)[STRUCTURED_COLUMNS]], axis=1)


def make_preprocessor(kind="both", k_premise=None, k_hypothesis=None, dense=False):
    """kind = "tfidf", "structured" or "both". k_* keeps only the top-k TF-IDF terms (chi-squared)."""

    def text_pipe(k):
        steps = [("tfidf", TfidfVectorizer(token_pattern=TOKEN_PATTERN, ngram_range=(1, 2),
                                           min_df=2, sublinear_tf=True))]
        if k is not None:
            steps.append(("select", SelectKBest(chi2, k=k)))
        return Pipeline(steps)

    parts = []
    if kind in ("tfidf", "both"):
        parts.append(("premise_tfidf", text_pipe(k_premise), "premise"))
        parts.append(("hypothesis_tfidf", text_pipe(k_hypothesis), "hypothesis"))
    if kind in ("structured", "both"):
        parts.append(("structured", StandardScaler(), STRUCTURED_COLUMNS))
    return ColumnTransformer(parts, sparse_threshold=0.0 if dense else 0.3)
