"""Structured features for every (premise, hypothesis) row (notebook 03)."""
import numpy as np
import pandas as pd

FEATURES = ["num_variables", "n_correlations", "n_independencies",
            "rel_collider", "rel_confounder", "rel_direct_cause", "rel_indirect_cause",
            "xy_correlated", "xy_independent", "xy_independent_given",
            "edge_x_to_y", "edge_y_to_x", "edge_undirected", "log_mec_size", "mec_support"]

RELATIONS = ["collider", "confounder", "direct_cause", "indirect_cause"]


def add_claim_type(data):
    """One-hot encode the claim type (no order between types, so not label encoding)."""
    data = data.copy()
    for relation in RELATIONS:
        data[f"rel_{relation}"] = (data["relation"] == relation).astype(int)
    return data


def add_premise_facts(data):
    """What the premise says overall and about the pair (x, y) in the hypothesis."""
    data = data.copy()
    data["n_correlations"] = data["premise"].str.count("correlates with")
    data["n_independencies"] = data["premise"].str.count("independent")

    def says(row, pattern):
        return pattern.format(a=row.x, b=row.y) in row.premise or pattern.format(a=row.y, b=row.x) in row.premise

    data["xy_correlated"] = data.apply(lambda r: says(r, "{a} correlates with {b}."), axis=1).astype(int)
    data["xy_independent"] = data.apply(lambda r: says(r, "{a} is independent of {b}."), axis=1).astype(int)
    data["xy_independent_given"] = data.apply(lambda r: says(r, "{a} and {b} are independent given"), axis=1).astype(int)
    return data


def add_graph_facts(data):
    """What the PC graph says about the pair: which way the x-y edge points, and how uncertain the graph is."""
    data = data.copy()
    data["edge_x_to_y"] = data.apply(lambda r: f"{r.x}->{r.y}" in r.directed_edges.split(";"), axis=1).astype(int)
    data["edge_y_to_x"] = data.apply(lambda r: f"{r.y}->{r.x}" in r.directed_edges.split(";"), axis=1).astype(int)
    data["edge_undirected"] = data.apply(
        lambda r: f"{min(r.x, r.y)}-{max(r.x, r.y)}" in r.undirected_edges.split(";"), axis=1).astype(int)
    data["log_mec_size"] = np.log(data["mec_size"])
    return data


def build_structured_features(data):
    """All 15 structured features (mec_support already comes from notebook 02)."""
    return add_graph_facts(add_premise_facts(add_claim_type(data)))
