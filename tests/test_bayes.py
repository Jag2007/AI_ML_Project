import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from c2c.bayes import build_dag, do_effect


def test_build_dag_skips_bad_edges():
    edges = pd.DataFrame({"x": ["A", "B", "C", "B", "A"], "y": ["B", "C", "A", "B", "G"]})
    dag = build_dag(["A", "B", "C"], edges)
    assert set(dag.edges) == {("A", "B"), ("B", "C")}   # C->A (cycle), B->B (self-loop), A->G (unknown) skipped


def test_do_effect_follows_the_arrows():
    model = DiscreteBayesianNetwork([("A", "B"), ("B", "C")])
    model.add_cpds(TabularCPD("A", 2, [[0.5], [0.5]]),
                   TabularCPD("B", 2, [[0.9, 0.2], [0.1, 0.8]], evidence=["A"], evidence_card=[2]),
                   TabularCPD("C", 2, [[0.9, 0.2], [0.1, 0.8]], evidence=["B"], evidence_card=[2]))
    # effect of A on C = 0.7 * 0.7 = 0.49, and forcing C cannot change A
    assert abs(do_effect(model, "A", "C") - 0.49) < 1e-9
    assert do_effect(model, "C", "A") == 0.0
