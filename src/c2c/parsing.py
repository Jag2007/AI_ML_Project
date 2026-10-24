"""Read Corr2Cause premises and hypotheses with regular expressions (notebook 02).

The text is generated from fixed templates, so regex reads it exactly.
"""
import re

import pandas as pd

ALLOWED_VARIABLES = list("ABCDEF")


def parse_variables(premise):
    """The variables listed in the premise, e.g. ['A', 'B', 'C']."""
    listed = re.search(r"variables, (.*?)\. All", premise).group(1)
    return listed.replace(" and ", ", ").split(", ")


def parse_independencies(premise):
    """Every independence fact in the premise as (x, y, given), where given is a list (empty if none)."""
    facts = [(x, y, []) for x, y in re.findall(r"([A-F]) is independent of ([A-F])\.", premise)]
    for x, y, given in re.findall(r"([A-F]) and ([A-F]) are independent given (.*?)\.", premise):
        facts.append((x, y, given.replace(" and ", ", ").split(", ")))
    return facts


def parse_hypothesis(df):
    """Add the two variables (x, y) and the claim type (relation) of every hypothesis."""
    df = df.copy()
    # the two variables are the single capital letters in the sentence
    letters = df["hypothesis"].str.findall(r"\b[A-Z]\b")
    df["x"] = letters.str[0]
    df["y"] = letters.str[1]

    # the claim type comes from the wording
    df["relation"] = "unknown"
    df.loc[df["hypothesis"].str.contains("directly causes"), "relation"] = "direct_cause"
    df.loc[df["hypothesis"].str.contains("something else which causes|but not a direct one"), "relation"] = "indirect_cause"
    df.loc[df["hypothesis"].str.contains("collider"), "relation"] = "collider"
    df.loc[df["hypothesis"].str.contains("confounder"), "relation"] = "confounder"

    # "Y is a cause for X, but not a direct one" names the cause first, so x is already the cause
    return df


def check_schema(df):
    """One column per rule, True where a row breaks it: only A-F, no self-loops, variables must be in the premise."""
    checks = pd.DataFrame(index=df.index)
    checks["outside A-F"] = ~df["x"].isin(ALLOWED_VARIABLES) | ~df["y"].isin(ALLOWED_VARIABLES)
    checks["self-loop"] = df["x"] == df["y"]
    checks["not in premise"] = ~df.apply(
        lambda row: str(row["x"]) in row["variables"] and str(row["y"]) in row["variables"], axis=1)
    checks["unknown wording"] = df["relation"] == "unknown"
    return checks
