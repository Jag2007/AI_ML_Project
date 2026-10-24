import networkx as nx

from c2c.graphs import all_dags, claim_is_true, find_graph, mec_support, path_relations, simple_claim_is_true

HEAD = "Suppose there is a closed system of 3 variables, A, B and C. All the statistical relations among these 3 variables are as follows: "


def test_pc_finds_a_collider():
    # A and B are independent but both correlate with C, so the only fitting graph is A -> C <- B
    pdag = find_graph(HEAD + "A correlates with C. B correlates with C. However, A is independent of B.")
    assert set(pdag.directed_edges) == {("A", "C"), ("B", "C")}
    assert len(all_dags(pdag)) == 1


def test_chain_has_three_fitting_dags():
    # A - B - C with A and C independent given B: A->B->C, A<-B<-C and A<-B->C all fit
    pdag = find_graph(HEAD + "A correlates with B. A correlates with C. B correlates with C. "
                             "However, A and C are independent given B.")
    dags = all_dags(pdag)
    assert len(dags) == 3
    assert mec_support(dags, "direct_cause", "A", "B") == 1 / 3


def test_indirect_cause_allows_a_direct_edge_too():
    # A -> B -> C and A -> C: "A causes something else which causes C" is true (through B)
    dag = nx.DiGraph([("A", "B"), ("B", "C"), ("A", "C")])
    assert claim_is_true(dag, "indirect_cause", "A", "C")
    assert not simple_claim_is_true(dag, "indirect_cause", "A", "C")


def test_collider_can_be_a_common_descendant():
    # B -> C -> E <- D: E is a common effect of B and D through a path, not a direct child of B
    dag = nx.DiGraph([("B", "C"), ("C", "E"), ("D", "E")])
    assert "collider" in path_relations(dag, "B", "D")
    assert claim_is_true(dag, "collider", "D", "B")


def test_confounder_quirk():
    # A <- C <- Z -> B: Z causes both, but it is not directly connected to B's side in the generator's check
    dag = nx.DiGraph([("Z", "C"), ("C", "A"), ("Z", "D"), ("D", "B")])
    assert "confounder" in path_relations(dag, "A", "B", confounder_quirk=False)
    assert "confounder" not in path_relations(dag, "A", "B", confounder_quirk=True)
    # with the common cause next to B, both versions agree
    dag = nx.DiGraph([("Z", "C"), ("C", "A"), ("Z", "B")])
    assert "confounder" in path_relations(dag, "A", "B", confounder_quirk=True)
