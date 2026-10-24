"""PC algorithm, Markov equivalence class and the claim rules (notebook 02).

The claim rules follow the Corr2Cause generator (code/data_gen.cpp, function compute_relation, in
https://github.com/causalNLP/corr2cause). It walks every simple path between X and Y in the skeleton and names
the path by its arrows. A claim is valid when it holds in every DAG of the equivalence class.
"""
import itertools

import networkx as nx
import pandas as pd
from pgmpy.base import DAG
from pgmpy.causal_discovery import PC
from pgmpy.independencies import Independencies

from c2c.parsing import parse_independencies, parse_variables


def find_graph(premise):
    """Run PC on the independences written in the premise and return the PDAG.

    PC normally tests independence on numbers. Here the premise already states the independences, so pgmpy reads
    them directly (ci_test="independence_match"). pgmpy still needs a table with the variable names, so we pass
    two rows of zeros that are never used.
    """
    variables = parse_variables(premise)
    independencies = Independencies()
    for x, y, given in parse_independencies(premise):
        independencies.add_assertions([x, y, given] if given else [x, y])

    pc = PC(variant="orig", ci_test="independence_match", show_progress=False)
    pc.fit(pd.DataFrame(0, index=[0, 1], columns=variables), independencies=independencies)
    return pc.causal_graph_


def all_dags(pdag):
    """Every DAG in the equivalence class: try both directions of each undirected edge and keep the ones that
    have no cycle and are I-equivalent (same skeleton and colliders) to the PC graph."""
    one_dag = pdag.to_dag()
    undirected = list(pdag.undirected_edges)
    dags = []
    for flips in itertools.product([False, True], repeat=len(undirected)):
        dag = DAG()
        dag.add_nodes_from(pdag.nodes)
        dag.add_edges_from(pdag.directed_edges)
        dag.add_edges_from([(b, a) if flip else (a, b) for (a, b), flip in zip(undirected, flips)])
        if nx.is_directed_acyclic_graph(dag) and dag.is_iequivalent(one_dag):
            dags.append(dag)
    return dags


def path_relations(dag, x, y, confounder_quirk=True):
    """Names of every simple path between x and y, following the Corr2Cause generator:

    * parent:     x -> y (one edge)
    * ancestor:   x -> ... -> y (two or more edges, all forward)
    * descendant: x <- ... <- y
    * collider:   x -> ... -> z <- ... <- y
    * confounder: x <- ... <- z -> ... -> y

    confounder_quirk=True copies a loop in the generator that runs i++ instead of i--. Because of it, a path only
    counts as a confounder when the common cause z is directly connected to y (x <- ... <- z -> y).
    """
    skeleton = dag.to_undirected()
    names = set()
    for path in nx.all_simple_paths(skeleton, x, y):
        steps = len(path) - 1
        if steps == 1:
            names.add("parent" if dag.has_edge(x, y) else "child")
            continue

        forward = 0                          # arrows pointing away from x, counted from the x end
        for i in range(steps):
            if not dag.has_edge(path[i], path[i + 1]):
                break
            forward += 1
        backward = 0                         # arrows pointing away from y, counted from the y end
        for i in range(steps, 0, -1):
            if not dag.has_edge(path[i], path[i - 1]):
                break
            backward += 1

        if forward == steps:
            names.add("ancestor")
        elif backward == steps:
            names.add("descendant")
        elif forward + backward == steps:
            names.add("collider")
        else:
            into_x = 0                       # arrows pointing towards x, counted from the x end
            for i in range(steps):
                if not dag.has_edge(path[i + 1], path[i]):
                    break
                into_x += 1
            if confounder_quirk:
                into_y = 1 if dag.has_edge(path[steps - 1], path[steps]) else 0
            else:
                into_y = 0                   # arrows pointing towards y, counted from the y end
                for i in range(steps, 0, -1):
                    if not dag.has_edge(path[i - 1], path[i]):
                        break
                    into_y += 1
            if into_x + into_y == steps:
                names.add("confounder")
    return names


def claim_is_true(dag, relation, x, y):
    """Is the claim true in this DAG? Uses the Corr2Cause generator's definitions."""
    if relation == "direct_cause":
        return dag.has_edge(x, y)
    if relation == "indirect_cause":
        # "x causes something else which causes y": a directed path of 2+ edges, even if x -> y also exists
        return "ancestor" in path_relations(dag, x, y)
    # the generator scores colliders and confounders on the pair in alphabetical order
    first, second = sorted([x, y])
    return relation in path_relations(dag, first, second)


def simple_claim_is_true(dag, relation, x, y):
    """Our first, simpler definitions, kept to show why they disagree with the labels."""
    if relation == "direct_cause":
        return dag.has_edge(x, y)
    if relation == "indirect_cause":
        return not dag.has_edge(x, y) and y in nx.descendants(dag, x)
    if relation == "collider":
        return len(set(dag.successors(x)) & set(dag.successors(y))) > 0
    if relation == "confounder":
        return len(set(dag.predecessors(x)) & set(dag.predecessors(y))) > 0


def mec_support(dags, relation, x, y, rule=claim_is_true):
    """Share of the DAGs in the equivalence class where the claim is true (1 means it must be true)."""
    return sum(rule(dag, relation, x, y) for dag in dags) / len(dags)
