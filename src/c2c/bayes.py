"""Build DAGs from predicted edges and measure intervention effects (notebook 05)."""
import networkx as nx
from pgmpy.inference import CausalInference


def build_dag(variables, edges):
    """Add edges one by one (most confident first). Skip self-loops, unknown variables and edges that would make
    a cycle, so the result is always a DAG."""
    dag = nx.DiGraph()
    dag.add_nodes_from(variables)
    for x, y in zip(edges["x"], edges["y"]):
        if x != y and x in variables and y in variables and not nx.has_path(dag, y, x):
            dag.add_edge(x, y)
    return dag


def do_effect(model, cause, effect):
    """P(effect=1 | do(cause=1)) - P(effect=1 | do(cause=0)) in a fitted pgmpy network."""
    # forcing a variable can't change anything that is not downstream of it
    if effect not in nx.descendants(nx.DiGraph(model.edges()), cause):
        return 0.0
    inference = CausalInference(model)
    p1 = inference.query([effect], do={cause: 1}, show_progress=False).values[1]
    p0 = inference.query([effect], do={cause: 0}, show_progress=False).values[1]
    return float(p1 - p0)
