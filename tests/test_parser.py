import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.parser import ParseError, parse_hypothesis, parse_premise  # noqa: E402

COLLIDER_PREMISE = (
    "Suppose there is a closed system of 3 variables, A, B and C. All the statistical relations "
    "among these 3 variables are as follows: A correlates with C. B correlates with C. "
    "However, A is independent of B."
)


def test_premise_variables_and_counts():
    facts = parse_premise(COLLIDER_PREMISE)
    assert facts.variables == ["A", "B", "C"]
    assert facts.correlations == {frozenset("AC"), frozenset("BC")}
    assert facts.independencies == [("A", "B", frozenset())]


def test_conditional_independence_with_several_variables():
    premise = (
        "Suppose there is a closed system of 4 variables, A, B, C and D. All the statistical "
        "relations among these 4 variables are as follows: A correlates with D. "
        "However, A and D are independent given B and C. B and C are independent given A, D."
    )
    facts = parse_premise(premise)
    assert ("A", "D", frozenset("BC")) in facts.independencies
    assert ("B", "C", frozenset("AD")) in facts.independencies
    assert facts.is_independent("D", "A", given={"C", "B"})
    assert not facts.is_independent("A", "D")


@pytest.mark.parametrize("text, relation, x, y", [
    ("A directly causes B.", "direct_cause", "A", "B"),
    ("B causes something else which causes C.", "indirect_cause", "B", "C"),
    ("C is a cause for A, but not a direct one.", "indirect_cause", "C", "A"),
    ("There exists at least one collider (i.e., common effect) of A and C.", "collider", "A", "C"),
    ("There exists at least one confounder (i.e., common cause) of B and C.", "confounder", "B", "C"),
])
def test_every_hypothesis_template(text, relation, x, y):
    hyp = parse_hypothesis(text)
    assert (hyp.relation, hyp.x, hyp.y) == (relation, x, y)


def test_extra_whitespace_is_tolerated():
    assert parse_hypothesis("  A   directly causes\nB. ").relation == "direct_cause"


def test_self_loop_hypothesis_is_rejected():
    with pytest.raises(ParseError, match="self-loop"):
        parse_hypothesis("A directly causes A.")


def test_self_loop_in_premise_is_rejected():
    with pytest.raises(ParseError, match="self-loop"):
        parse_premise(COLLIDER_PREMISE + " A correlates with A.")


def test_variable_outside_schema_is_rejected():
    with pytest.raises(ParseError, match="outside the A-F schema"):
        parse_hypothesis("A directly causes G.")


def test_variable_not_declared_in_premise_is_rejected():
    with pytest.raises(ParseError, match="not declared"):
        parse_hypothesis("A directly causes D.", variables=["A", "B", "C"])


def test_variable_count_mismatch_is_rejected():
    with pytest.raises(ParseError):
        parse_premise(COLLIDER_PREMISE.replace("3 variables, A, B and C", "4 variables, A, B and C", 1))


def test_unknown_hypothesis_is_rejected():
    with pytest.raises(ParseError, match="template"):
        parse_hypothesis("A is friends with B.")

