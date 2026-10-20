"""PC algorithm that uses the premise's independence facts instead of statistical tests."""

from itertools import combinations, product

import networkx as nx


class CPDAG:
    """Directed edges = direction is certain. Undirected edges = direction can't be known."""

    def __init__(self, nodes, directed, undirected):
        self.nodes = list(nodes)
        self.directed = set(directed)                         # {(X, Y)} means X -> Y
        self.undirected = {frozenset(e) for e in undirected}  # {frozenset({X, Y})}

    def adjacent(self, x, y):
        return (x, y) in self.directed or (y, x) in self.directed or frozenset((x, y)) in self.undirected

    def __repr__(self):
        d = ", ".join(f"{a}->{b}" for a, b in sorted(self.directed))
        u = ", ".join("-".join(sorted(e)) for e in sorted(self.undirected, key=sorted))
        return f"CPDAG(directed=[{d}], undirected=[{u}])"


def pc_skeleton(facts):
    # Start fully connected, then remove every edge the premise says is independent
    nodes = facts.variables
    adj = {v: set(nodes) - {v} for v in nodes}
    sepsets = {}
    size = 0
    while any(len(adj[x]) - 1 >= size for x in nodes):
        for x in nodes:
            for y in sorted(adj[x]):
                if y not in adj[x]:
                    continue
                for cond in combinations(sorted(adj[x] - {y}), size):
                    if facts.is_independent(x, y, cond):
                        adj[x].discard(y)
                        adj[y].discard(x)
                        sepsets[frozenset((x, y))] = frozenset(cond)
                        break
        size += 1
    return adj, sepsets


def _orient_v_structures(nodes, adj, sepsets):
    # X - Z - Y with X, Y not joined and Z not in their separating set  =>  X -> Z <- Y
    directed = set()
    for z in nodes:
        for x, y in combinations(sorted(adj[z]), 2):
            if y not in adj[x] and z not in sepsets.get(frozenset((x, y)), frozenset()):
                directed.add((x, z))
                directed.add((y, z))
    return directed


def _apply_meek_rules(nodes, adj, directed):
    def is_undirected(a, b):
        return b in adj[a] and (a, b) not in directed and (b, a) not in directed

    changed = True
    while changed:
        changed = False
        for a, b in product(nodes, repeat=2):
            if a == b or not is_undirected(a, b):
                continue
            # R1: c -> a - b, c and b not joined  =>  a -> b
            r1 = any((c, a) in directed and b not in adj[c] and c != b for c in nodes)
            # R2: a -> c -> b  =>  a -> b (otherwise a cycle)
            r2 = any((a, c) in directed and (c, b) in directed for c in nodes)
            # R3: a - c -> b and a - d -> b, c and d not joined  =>  a -> b
            r3 = any(
                is_undirected(a, c) and is_undirected(a, d)
                and (c, b) in directed and (d, b) in directed and d not in adj[c]
                for c, d in combinations(nodes, 2)
            )
            if r1 or r2 or r3:
                directed.add((a, b))
                changed = True
    return directed


def run_pc(facts):
    adj, sepsets = pc_skeleton(facts)
    directed = _orient_v_structures(facts.variables, adj, sepsets)
    directed = _apply_meek_rules(facts.variables, adj, directed)
    undirected = {
        frozenset((x, y)) for x in facts.variables for y in adj[x]
        if (x, y) not in directed and (y, x) not in directed
    }
    return CPDAG(facts.variables, directed, undirected)


def _v_structures(g):
    return {
        (min(x, y), z, max(x, y))
        for z in g.nodes for x, y in combinations(g.predecessors(z), 2)
        if not g.has_edge(x, y) and not g.has_edge(y, x)
    }


def markov_equivalence_class(cpdag):
    """Every DAG that fits the CPDAG (brute force is fine for at most 6 variables)."""
    base = nx.DiGraph()
    base.add_nodes_from(cpdag.nodes)
    base.add_edges_from(cpdag.directed)
    target = {(x, z, y) for x, z, y in _v_structures(base) if not cpdag.adjacent(x, y)}
    undirected = [tuple(sorted(e)) for e in cpdag.undirected]
    members = []
    for flips in product((False, True), repeat=len(undirected)):
        g = base.copy()
        g.add_edges_from((b, a) if flip else (a, b) for (a, b), flip in zip(undirected, flips))
        if nx.is_directed_acyclic_graph(g) and _v_structures(g) == target:
            members.append(g)
    return members


def relation_holds(g, relation, x, y):
    if relation == "direct_cause":
        return g.has_edge(x, y)
    if relation == "indirect_cause":
        return not g.has_edge(x, y) and y in nx.descendants(g, x)
    if relation == "collider":
        return any(g.has_edge(x, z) and g.has_edge(y, z) for z in g.nodes)
    if relation == "confounder":
        return any(g.has_edge(z, x) and g.has_edge(z, y) for z in g.nodes)
    raise ValueError(relation)


def relation_support(members, relation, x, y):
    """Share of compatible DAGs where the claim is true (1.0 means it must be true)."""
    return sum(relation_holds(g, relation, x, y) for g in members) / len(members)
