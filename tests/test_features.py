import pandas as pd

from c2c.features import FEATURES, build_structured_features


def test_one_row():
    row = pd.DataFrame([{
        "premise": "Suppose ... A correlates with C. B correlates with C. However, A is independent of B.",
        "relation": "collider", "x": "A", "y": "B", "num_variables": 3,
        "directed_edges": "A->C;B->C", "undirected_edges": "", "mec_size": 1, "mec_support": 1.0,
    }])
    out = build_structured_features(row).iloc[0]
    assert set(FEATURES) <= set(out.index)
    assert out["rel_collider"] == 1 and out["rel_direct_cause"] == 0
    assert out["n_correlations"] == 2 and out["n_independencies"] == 1
    assert out["xy_correlated"] == 0 and out["xy_independent"] == 1 and out["xy_independent_given"] == 0
    assert out["edge_x_to_y"] == 0 and out["edge_y_to_x"] == 0 and out["edge_undirected"] == 0
    assert out["log_mec_size"] == 0


def test_edge_direction():
    row = pd.DataFrame([{
        "premise": "x", "relation": "direct_cause", "x": "B", "y": "A", "num_variables": 2,
        "directed_edges": "A->B", "undirected_edges": "", "mec_size": 1, "mec_support": 0.0,
    }])
    out = build_structured_features(row).iloc[0]
    assert out["edge_x_to_y"] == 0 and out["edge_y_to_x"] == 1
