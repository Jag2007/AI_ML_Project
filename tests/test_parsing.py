import pandas as pd

from c2c.parsing import check_schema, parse_hypothesis, parse_independencies, parse_variables

PREMISE = ("Suppose there is a closed system of 3 variables, A, B and C. All the statistical relations among these "
           "3 variables are as follows: A correlates with C. B correlates with C. However, A is independent of B. "
           "A and B are independent given D and E.")


def test_variables():
    assert parse_variables(PREMISE) == ["A", "B", "C"]


def test_independencies():
    assert parse_independencies(PREMISE) == [("A", "B", []), ("A", "B", ["D", "E"])]


def test_every_claim_type():
    df = pd.DataFrame({"hypothesis": [
        "A directly causes C.",
        "A causes something else which causes C.",
        "C is a cause for A, but not a direct one.",
        "There exists at least one collider (i.e., common effect) of A and B.",
        "There exists at least one confounder (i.e., common cause) of B and C.",
    ]})
    out = parse_hypothesis(df)
    assert list(out["relation"]) == ["direct_cause", "indirect_cause", "indirect_cause", "collider", "confounder"]
    assert list(out["x"]) == ["A", "A", "C", "A", "B"]
    assert list(out["y"]) == ["C", "C", "A", "B", "C"]


def test_schema_catches_bad_rows():
    df = parse_hypothesis(pd.DataFrame({
        "hypothesis": ["A directly causes B.", "C directly causes C.", "A directly causes G.", "A and B are friends."],
        "variables": ["A, B, C"] * 4,
    }))
    checks = check_schema(df)
    assert not checks.iloc[0].any()
    assert checks.loc[1, "self-loop"]
    assert checks.loc[2, "outside A-F"] and checks.loc[2, "not in premise"]
    assert checks.loc[3, "unknown wording"]
