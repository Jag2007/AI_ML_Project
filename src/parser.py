"""Regex parser: turns Corr2Cause premises and hypotheses into structured facts."""

import re
from dataclasses import dataclass, field

# Entity schema: Corr2Cause only ever uses the variables A to F
VARIABLES = ("A", "B", "C", "D", "E", "F")

_VAR = r"[A-Z]"
_SYSTEM_RE = re.compile(r"closed system of (\d+) variables?, ([A-Z, and]+?)\.")
_CORR_RE = re.compile(rf"\b({_VAR}) correlates with ({_VAR})\.")
_MARGINAL_INDEP_RE = re.compile(rf"\b({_VAR}) is independent of ({_VAR})\.")
_COND_INDEP_RE = re.compile(rf"\b({_VAR}) and ({_VAR}) are independent given ([A-Z, and]+?)\.")

# "non-parent ancestor" and "non-child descendant" are two wordings of the same claim (indirect cause)
_HYPOTHESIS_PATTERNS = [
    (re.compile(rf"^({_VAR}) directly causes ({_VAR})\.$"), "direct_cause"),
    (re.compile(rf"^({_VAR}) causes something else which causes ({_VAR})\.$"), "indirect_cause"),
    (re.compile(rf"^({_VAR}) is a cause for ({_VAR}), but not a direct one\.$"), "indirect_cause"),
    (re.compile(rf"^There exists at least one collider \(i\.e\., common effect\) of ({_VAR}) and ({_VAR})\.$"), "collider"),
    (re.compile(rf"^There exists at least one confounder \(i\.e\., common cause\) of ({_VAR}) and ({_VAR})\.$"), "confounder"),
]

HYPOTHESIS_TYPES = ("direct_cause", "indirect_cause", "collider", "confounder")


class ParseError(ValueError):
    """Raised when a sentence does not fit the schema (unknown variable, self-loop, ...)."""


@dataclass
class PremiseFacts:
    variables: list
    correlations: set = field(default_factory=set)      # {frozenset({X, Y})}
    independencies: list = field(default_factory=list)  # [(X, Y, frozenset(conditioning set))]

    @property
    def num_variables(self):
        return len(self.variables)

    def is_independent(self, x, y, given=frozenset()):
        """Oracle used by the PC algorithm: True only if the premise states it."""
        given = frozenset(given)
        return any({a, b} == {x, y} and cond == given for a, b, cond in self.independencies)


@dataclass
class HypothesisFacts:
    relation: str
    x: str
    y: str


def clean_text(text):
    """Collapse repeated whitespace so the regexes never trip over stray spaces or newlines."""
    return re.sub(r"\s+", " ", str(text)).strip()


def _variable_list(chunk):
    # "A, B, C and D" -> ["A", "B", "C", "D"]
    return [v.strip() for v in re.split(r",|\band\b", chunk) if v.strip()]


def _check_variable(v, allowed):
    if v not in VARIABLES:
        raise ParseError(f"variable {v!r} is outside the A-F schema")
    if v not in allowed:
        raise ParseError(f"variable {v!r} is not declared in the premise")


def parse_premise(premise):
    premise = clean_text(premise)
    m = _SYSTEM_RE.search(premise)
    if not m:
        raise ParseError("could not find the 'closed system of N variables' sentence")
    declared = int(m.group(1))
    variables = _variable_list(m.group(2))
    if len(variables) != declared:
        raise ParseError(f"premise says {declared} variables but lists {variables}")
    for v in variables:
        _check_variable(v, variables)

    facts = PremiseFacts(variables=variables)
    for x, y in _CORR_RE.findall(premise):
        for v in (x, y):
            _check_variable(v, variables)
        if x == y:
            raise ParseError(f"self-loop in correlation: {x} correlates with {y}")
        facts.correlations.add(frozenset((x, y)))

    for x, y in _MARGINAL_INDEP_RE.findall(premise):
        for v in (x, y):
            _check_variable(v, variables)
        if x == y:
            raise ParseError(f"self-loop in independence: {x} independent of {y}")
        facts.independencies.append((x, y, frozenset()))

    for x, y, given in _COND_INDEP_RE.findall(premise):
        cond = _variable_list(given)
        for v in (x, y, *cond):
            _check_variable(v, variables)
        if x == y or x in cond or y in cond:
            raise ParseError(f"self-loop in conditional independence: {x}, {y} | {cond}")
        facts.independencies.append((x, y, frozenset(cond)))
    return facts


def parse_hypothesis(hypothesis, variables=VARIABLES):
    hypothesis = clean_text(hypothesis)
    for pattern, relation in _HYPOTHESIS_PATTERNS:
        m = pattern.match(hypothesis)
        if m:
            x, y = m.groups()
            for v in (x, y):
                _check_variable(v, variables)
            if x == y:
                raise ParseError(f"self-loop hypothesis: {hypothesis}")
            return HypothesisFacts(relation=relation, x=x, y=y)
    raise ParseError(f"hypothesis does not match any known template: {hypothesis}")
