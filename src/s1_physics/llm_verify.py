"""
src/s1_physics/llm_verify.py

The verification half of the S1 LLM reporting experiment (protocol 10.1).

The task is to check whether an LLM can turn a structured experimental record
into a scientific summary without inventing, altering or misreporting numbers.
The checking must be deterministic Python, not a second language model. An
LLM judge would inherit the failure mode it is meant to detect, and it cannot
be inspected during the code demonstration. Everything here is arithmetic and
string matching, and every verdict can be traced to a rule.

Section 10.1 requires the procedure to identify at least:
    numbers not present in the source        -> UNSUPPORTED
    changed numerical values                 -> ALTERED (near a source value
                                                but outside rounding tolerance)
    unsupported comparisons                  -> comparative claim checks
    incorrect ranking statements             -> superlative claim checks
    incorrect percentages                    -> DERIVED check on ratios,
                                                differences and percentages
    invented experimental conclusions        -> claims about quantities the
                                                record does not contain

Verdicts for a numeric claim, strongest first:

    PROMPT_ECHO  a number that came from the instructions rather than the
                 record. Models routinely open with "Here is a 150-word
                 summary", echoing the length limit in the prompt. That is not
                 a claim about the results and must not count against the
                 model. Before this category existed it was flagged ALTERED,
                 because 150 sits within tolerance of a runtime of 161.4
                 seconds, and it failed every single run of one experiment
    SUPPORTED    equals a value in the record, allowing for the rounding the
                 text itself displays (0.88 matches 0.8771)
    DERIVED      not stored directly but computable from stored values: a
                 difference, a ratio, a percentage, or a percentage change
    ALTERED      close to a source value but wrong beyond rounding. This is
                 the dangerous category, since it reads as authoritative
    UNSUPPORTED  no relation to anything in the record

Usage:
    from llm_verify import build_record, verify
    record = build_record()
    report = verify(summary_text, record)
    print(report.as_text())
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))
from config import ARTIFACT_DIR

# Numbers written with these many significant decimals or fewer are treated as
# deliberately rounded rather than altered.
MAX_ROUNDING_DECIMALS = 6
# A claimed number within this relative distance of a source value, but not
# explainable by rounding, is ALTERED rather than UNSUPPORTED.
ALTERED_REL_TOL = 0.25

MODEL_ALIASES = {
    "svm": ("svm", "rbf-svm", "rbf svm", "support vector"),
    "xgb": ("xgboost", "xgb", "gradient boosting"),
    "rf": ("random forest", "randomforest", "rf"),
    "mlp": ("mlp", "multilayer perceptron", "multi-layer perceptron",
            "neural network"),
}

METRIC_ALIASES = {
    "mean_macro_f1": ("macro-f1", "macro f1", "macrof1", "f1"),
    "accuracy": ("accuracy",),
    "ece": ("ece", "expected calibration error", "calibration error"),
    "brier_score": ("brier", "brier score"),
    "wall_seconds": ("time", "runtime", "wall time", "training time"),
    "parameter_count": ("parameter", "parameters"),
}

# Higher is better for these; lower is better for the rest.
HIGHER_IS_BETTER = {"mean_macro_f1", "accuracy"}

# Quality words depend on whether the metric is a score or a loss.
QUALITY_BEST = ("best", "strongest", "leading", "most accurate",
                "outperforms all", "wins")
QUALITY_WORST = ("worst", "weakest", "poorest", "least accurate")
# Direction words refer to the number itself, whatever the metric means.
# "the lowest ECE" is a claim about the value, not about quality, so it must
# be checked against the minimum regardless of the metric's polarity.
DIRECTION_HIGH = ("highest", "largest", "greatest", "top", "slowest",
                  "longest")
DIRECTION_LOW = ("lowest", "smallest", "least", "fastest", "quickest",
                 "shortest")

# Clause boundaries. A sentence such as "XGBoost followed with a macro-F1 of
# 0.8625, while offering the fastest training time and the lowest Brier score"
# contains three metrics and two superlatives, each belonging to a different
# clause. Scanning the whole sentence attached "lowest" to macro-F1 and
# reported a correct summary as a ranking error.
CLAUSE_RE = re.compile(
    r"\s*(?:,|;|:|\bwhile\b|\bwhereas\b|\bthough\b|\balthough\b"
    r"|\bbut\b|\bhowever\b|\band\b)\s*", re.IGNORECASE)
COMPARATIVE = ("better than", "higher than", "outperforms", "exceeds",
               "beats", "worse than", "lower than", "underperforms")


# --------------------------------------------------------------------------
# Building the source record
# --------------------------------------------------------------------------

def build_record(artifact_dir: Path | None = None) -> dict:
    """
    Assemble the structured record the LLM will be given, in the shape
    protocol section 10.1 specifies.
    """
    d = artifact_dir or ARTIFACT_DIR
    models = {}

    p = d / "results_classical_full.json"
    if p.exists():
        raw = json.loads(p.read_text())
        for key, m in raw["models"].items():
            models[key] = _model_entry(m)
        majority = raw.get("majority_baseline", {})
    else:
        majority = {}

    p = d / "results_mlp.json"
    if p.exists():
        models["mlp"] = _model_entry(json.loads(p.read_text())["model"])

    return {"task": "four-class micro-Doppler classification",
            "n_segments": 75801, "n_measurements": 130, "n_outer_folds": 5,
            "n_inner_folds": 5, "majority_baseline_macro_f1":
                round(majority.get("mean_macro_f1", 0.0), 4),
            "models": models}


def _model_entry(m: dict) -> dict:
    out = {
        "model_name": m["model_name"],
        "outer_fold_scores": [round(v, 4) for v in m["outer_fold_scores"]],
        "mean_macro_f1": round(m["mean_macro_f1"], 4),
        "std_macro_f1": round(m["std_macro_f1"], 4),
        "accuracy": round(m["accuracy"], 4),
        "ece": round(m["ece"], 4),
        "brier_score": round(m["brier_score"], 4),
        "wall_seconds": m["wall_seconds"],
    }
    if m.get("parameter_count"):
        out["parameter_count"] = m["parameter_count"]
    return out


# --------------------------------------------------------------------------
# Flattening and derived values
# --------------------------------------------------------------------------

def flatten(obj, prefix: str = "") -> dict[str, float]:
    """Every numeric leaf in the record, keyed by dotted path."""
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)
    return out


_METRIC_GROUPS = (
    ("macro_f1", ("mean_macro_f1", "outer_fold_scores",
                  "majority_baseline_macro_f1", "std_macro_f1")),
    ("accuracy", ("accuracy",)),
    ("ece", ("ece",)),
    ("brier", ("brier_score",)),
    ("seconds", ("wall_seconds",)),
    ("params", ("parameter_count",)),
    ("counts", ("n_segments", "n_measurements", "n_outer_folds",
                "n_inner_folds")),
)


def metric_group(path: str) -> str | None:
    """Which quantity a flattened path belongs to, for pairing purposes."""
    bare = re.sub(r"\[\d+\]$", "", path)
    for group, keys in _METRIC_GROUPS:
        if any(bare.endswith(k) or bare == k for k in keys):
            return group
    return None


def derived_values(facts: dict[str, float]) -> dict[float, str]:
    """
    Quantities a careful writer could legitimately compute from the record.

    Deliberately narrow. An earlier version combined every pair of facts with
    differences, ratios and percentage changes, which produced tens of
    thousands of candidate values and blanketed the number line: an invented
    macro-F1 of 0.8871 was "explained" as the ratio of two Brier scores. A
    verifier that can explain anything detects nothing.

    Pairs are therefore formed only WITHIN a metric group, so comparing two
    models on macro-F1 is allowed and dividing an accuracy by a runtime is
    not. Unit conversions are attached only to the quantities they apply to.
    """
    out: dict[float, str] = {}
    for k, v in facts.items():
        g = metric_group(k)
        if g in ("macro_f1", "accuracy", "ece", "brier"):
            out.setdefault(round(v * 100, 6), f"{k} as a percentage")
        if g == "seconds":
            out.setdefault(round(v / 60, 6), f"{k} in minutes")
            out.setdefault(round(v / 3600, 6), f"{k} in hours")

    # Individual fold scores are checkable directly and are excluded from
    # pairing: 20 of them generate hundreds of spurious differences.
    by_group: dict[str, list[tuple[str, float]]] = {}
    for k, v in facts.items():
        if "outer_fold_scores" in k:
            continue
        g = metric_group(k)
        if g:
            by_group.setdefault(g, []).append((k, v))

    # Ratios are only meaningful for quantities people actually express as
    # factors: runtimes and parameter counts. A ratio of two Brier scores is
    # not a quantity anyone reports, but it lands in the same 0-1 range as
    # macro-F1 and would launder an invented score as "derived".
    RATIO_OK = {"seconds", "params"}
    PCT_CHANGE_OK = {"macro_f1", "accuracy", "seconds"}

    for g, items in by_group.items():
        for i, (ka, va) in enumerate(items):
            for kb, vb in items[i + 1:]:
                out.setdefault(round(va - vb, 6), f"{ka} minus {kb}")
                out.setdefault(round(vb - va, 6), f"{kb} minus {ka}")
                if g in RATIO_OK and abs(vb) > 1e-12:
                    out.setdefault(round(va / vb, 6), f"{ka} over {kb}")
                if g in RATIO_OK and abs(va) > 1e-12:
                    out.setdefault(round(vb / va, 6), f"{kb} over {ka}")
                if g in PCT_CHANGE_OK and abs(vb) > 1e-12:
                    out.setdefault(round((va - vb) / vb * 100, 6),
                                   f"percent change from {kb} to {ka}")
                if g in PCT_CHANGE_OK and abs(va) > 1e-12:
                    out.setdefault(round((vb - va) / va * 100, 6),
                                   f"percent change from {ka} to {kb}")
    return out


# --------------------------------------------------------------------------
# Claim extraction
# --------------------------------------------------------------------------

NUMBER_RE = re.compile(
    r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.\d+|-?\d+)"
    r"\s*(%|x\b|percent)?", re.IGNORECASE)


@dataclass
class NumericClaim:
    text: str
    value: float
    suffix: str
    decimals: int
    sentence: str
    verdict: str = "UNSUPPORTED"
    matched: str = ""


@dataclass
class StatementClaim:
    sentence: str
    kind: str
    verdict: str
    detail: str


@dataclass
class Report:
    numeric: list[NumericClaim] = field(default_factory=list)
    statements: list[StatementClaim] = field(default_factory=list)
    # Clauses that named a model and a superlative but more than one metric,
    # and were therefore skipped rather than guessed at. Reported so the
    # checked proportion can be stated honestly rather than implied to be 100%.
    unchecked_clauses: int = 0

    @property
    def n_numbers(self) -> int:
        return len(self.result_claims)

    @property
    def result_claims(self) -> list:
        """Numbers that are claims about the results, excluding prompt echo."""
        return [c for c in self.numeric if c.verdict != "PROMPT_ECHO"]

    @property
    def factual_accuracy(self) -> float:
        claims = self.result_claims
        if not claims:
            return float("nan")
        ok = sum(c.verdict in ("SUPPORTED", "DERIVED") for c in claims)
        return ok / len(claims)

    @property
    def unsupported_rate(self) -> float:
        checkable = [s for s in self.statements if s.verdict != "UNCHECKED"]
        total = len(self.result_claims) + len(checkable)
        if not total:
            return float("nan")
        bad = sum(c.verdict in ("UNSUPPORTED", "ALTERED")
                  for c in self.result_claims)
        bad += sum(s.verdict == "INCORRECT" for s in self.statements)
        return bad / total

    def counts(self) -> dict:
        c = {v: 0 for v in ("SUPPORTED", "DERIVED", "ALTERED", "UNSUPPORTED",
                            "PROMPT_ECHO")}
        for n in self.numeric:
            c[n.verdict] += 1
        c["statements_correct"] = sum(s.verdict == "CORRECT"
                                      for s in self.statements)
        c["statements_incorrect"] = sum(s.verdict == "INCORRECT"
                                        for s in self.statements)
        c["statements_unchecked"] = sum(s.verdict == "UNCHECKED"
                                        for s in self.statements)
        return c

    def passed(self) -> bool:
        return (all(n.verdict in ("SUPPORTED", "DERIVED", "PROMPT_ECHO")
                    for n in self.numeric)
                and all(s.verdict in ("CORRECT", "UNCHECKED")
                        for s in self.statements))

    def as_text(self) -> str:
        echo = self.counts()["PROMPT_ECHO"]
        lines = [f"result claims: {self.n_numbers}"
                 + (f" (+{echo} echoed from the prompt)" if echo else "") + " | "
                 f"factual accuracy: {self.factual_accuracy:.3f} | "
                 f"unsupported-claim rate: {self.unsupported_rate:.3f}",
                 f"counts: {self.counts()}"]
        for c in self.numeric:
            if c.verdict in ("ALTERED", "UNSUPPORTED"):
                lines.append(f"  [{c.verdict}] '{c.text}' in: {c.sentence[:90]}")
        for s in self.statements:
            if s.verdict == "INCORRECT":
                lines.append(f"  [CLAIM {s.kind}] {s.detail}")
        lines.append("VERDICT: " + ("PASS" if self.passed() else "FAIL"))
        return "\n".join(lines)


# A period followed immediately by a capital letter, with no space, is a
# sentence boundary that generated text produces routinely. Leaving it
# unsplit let one sentence absorb the next one's subject: Gemini wrote
# "...the lowest Brier score (0.1091).Random Forest and MLP achieved...", and
# the clause carrying "lowest" picked up Random Forest as its model. The
# lookbehind excludes digits so decimals such as 0.8771 are untouched.
_MISSING_SPACE = re.compile(r"(?<=[^\d\s])\.(?=[A-Z])")


def _sentences(text: str) -> list[str]:
    text = _MISSING_SPACE.sub(". ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _classify_number(value: float, decimals: int, facts: dict[str, float],
                     derived: dict[float, str]) -> tuple[str, str]:
    # exact, or exact at the precision the text chose to display
    for path, v in facts.items():
        if v == value:
            return "SUPPORTED", path
        if decimals <= MAX_ROUNDING_DECIMALS and round(v, decimals) == value:
            return "SUPPORTED", f"{path} rounded to {decimals} dp"
    for dv, why in derived.items():
        if dv == value or (decimals <= MAX_ROUNDING_DECIMALS
                           and round(dv, decimals) == value):
            return "DERIVED", why
    # near a source value but not reachable by rounding: an altered figure
    best, best_path = None, ""
    for path, v in facts.items():
        if abs(v) < 1e-12:
            continue
        rel = abs(value - v) / abs(v)
        if best is None or rel < best:
            best, best_path = rel, path
    if best is not None and best <= ALTERED_REL_TOL:
        return "ALTERED", f"closest source value is {best_path}"
    return "UNSUPPORTED", ""


def _model_in(text: str) -> list[str]:
    """
    Models mentioned, ordered by where they appear in the sentence.

    Dictionary order would make "Random Forest outperforms the SVM" parse as
    a claim about the SVM outperforming Random Forest, inverting the check.
    """
    low = text.lower()
    found = []
    for k, aliases in MODEL_ALIASES.items():
        pos = min((low.find(a) for a in aliases if a in low), default=-1)
        if pos >= 0:
            found.append((pos, k))
    return [k for _, k in sorted(found)]


def _metric_in(text: str) -> str | None:
    low = text.lower()
    for metric, aliases in METRIC_ALIASES.items():
        if any(a in low for a in aliases):
            return metric
    return None


def _metrics_in(text: str) -> list[str]:
    """Every distinct metric the text mentions."""
    low = text.lower()
    found = []
    for metric, aliases in METRIC_ALIASES.items():
        if any(a in low for a in aliases):
            found.append(metric)
    return found


def _check_statements(text: str, record: dict) -> list[StatementClaim]:
    """
    Claims are evaluated per CLAUSE, not per sentence, and a clause naming
    more than one metric is skipped as ambiguous rather than guessed at. A
    verifier that reports false alarms is worse than useless here, because the
    whole point is to measure someone else's error rate.
    """
    out: list[StatementClaim] = []
    models = record.get("models", {})
    for sent in _sentences(text):
        subject: list[str] = []
        for clause in CLAUSE_RE.split(sent):
            if not clause.strip():
                continue
            out.extend(_check_clause(clause, subject, models))
    return out


def _check_clause(clause: str, subject: list[str],
                  models: dict) -> list[StatementClaim]:
    out: list[StatementClaim] = []
    low = clause.lower()
    here = _model_in(clause)
    if here:
        subject[:] = here                       # carries into later clauses
    named = here or subject
    metrics = _metrics_in(clause)
    if len(metrics) != 1 or not named:
        if len(metrics) > 1 and named:
            out.append(StatementClaim(clause, "ambiguous", "UNCHECKED",
                                      "clause names several metrics"))
        return out
    metric = metrics[0]
    sent = clause
    available = {k: m[metric] for k, m in models.items() if metric in m}
    if len(available) < 2:
        return out
    higher_better = metric in HIGHER_IS_BETTER

    # A direction word ("lowest ECE") refers to the number; a quality word
    # ("best") refers to the metric's meaning. For a loss like ECE these point
    # in opposite directions, so they cannot share a branch.
    wants_max = None
    kind = ""
    if any(w in low for w in DIRECTION_HIGH):
        wants_max, kind = True, "direction"
    elif any(w in low for w in DIRECTION_LOW):
        wants_max, kind = False, "direction"
    elif any(w in low for w in QUALITY_BEST):
        wants_max, kind = higher_better, "quality"
    elif any(w in low for w in QUALITY_WORST):
        wants_max, kind = not higher_better, "quality"

    if wants_max is not None:
        target = (max if wants_max else min)(available, key=available.get)
        ok = named[0] == target
        word = ("highest" if wants_max else "lowest") if kind == "direction" \
            else ("best" if wants_max == higher_better else "worst")
        out.append(StatementClaim(
            sent, f"superlative/{kind}", "CORRECT" if ok else "INCORRECT",
            f"claims {named[0]} has the {word} {metric}; record says "
            f"{target} ({available[target]}), while {named[0]} has "
            f"{available.get(named[0])}" if not ok else ""))
    elif len(named) >= 2 and any(w in low for w in COMPARATIVE):
        a, b = named[0], named[1]
        if a in available and b in available:
            worse_word = any(w in low for w in
                             ("worse than", "lower than", "underperforms"))
            claim_a_higher = not worse_word
            actual_a_higher = available[a] > available[b]
            ok = (claim_a_higher == actual_a_higher) if higher_better \
                else (claim_a_higher != actual_a_higher)
            out.append(StatementClaim(
                sent, "comparison", "CORRECT" if ok else "INCORRECT",
                f"claims {a} vs {b} on {metric}; record has "
                f"{a}={available[a]}, {b}={available[b]}" if not ok else ""))
    return out


def prompt_numbers(prompt: str | None) -> set[float]:
    """Numbers that appear in the instructions given to the model."""
    if not prompt:
        return set()
    return {float(m.group(1).replace(",", ""))
            for m in NUMBER_RE.finditer(prompt)}


def verify(text: str, record: dict, prompt: str | None = None) -> Report:
    """
    Check every numeric and comparative claim in `text` against `record`.

    Pass `prompt` so that numbers echoed from the instructions are separated
    from claims about the results. Without it, a model opening with "Here is a
    150-word summary" is charged with fabricating a figure.
    """
    facts = flatten(record)
    derived = derived_values(facts)
    echoes = prompt_numbers(prompt)
    report = Report()

    for sent in _sentences(text):
        for m in NUMBER_RE.finditer(sent):
            raw, suffix = m.group(1), (m.group(2) or "").lower()
            value = float(raw.replace(",", ""))
            if suffix in ("%", "percent"):
                pass                      # compared against percentage forms
            decimals = len(raw.split(".")[1]) if "." in raw else 0
            verdict, matched = _classify_number(value, decimals, facts, derived)
            if verdict in ("ALTERED", "UNSUPPORTED") and value in echoes:
                verdict, matched = "PROMPT_ECHO", "appears in the prompt"
            report.numeric.append(NumericClaim(
                text=m.group(0).strip(), value=value, suffix=suffix,
                decimals=decimals, sentence=sent, verdict=verdict,
                matched=matched))

    report.statements = _check_statements(text, record)
    return report


if __name__ == "__main__":
    rec = build_record()
    print(json.dumps(rec, indent=2))
    print(f"\n{len(flatten(rec))} numeric facts in the record")
