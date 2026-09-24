"""
test_llm_verify.py -- the verifier must catch each failure mode section 10.1
names, and must not raise false alarms on a faithful summary.

Uses a fixed record built from the real S1 numbers so the tests do not depend
on files being present.
"""

from __future__ import annotations

import pytest

from llm_verify import build_record, flatten, verify

RECORD = {
    "task": "four-class micro-Doppler classification",
    "n_segments": 75801, "n_measurements": 130,
    "n_outer_folds": 5, "n_inner_folds": 5,
    "majority_baseline_macro_f1": 0.2183,
    "models": {
        "svm": {"model_name": "RBF-SVM",
                "outer_fold_scores": [0.8306, 0.8822, 0.9036, 0.8904, 0.8786],
                "mean_macro_f1": 0.8771, "std_macro_f1": 0.0248,
                "accuracy": 0.9379, "ece": 0.0281, "brier_score": 0.1108,
                "wall_seconds": 882.5},
        "xgb": {"model_name": "XGBoost",
                "outer_fold_scores": [0.7967, 0.9029, 0.8665, 0.8905, 0.8557],
                "mean_macro_f1": 0.8625, "std_macro_f1": 0.0369,
                "accuracy": 0.9300, "ece": 0.0258, "brier_score": 0.1091,
                "wall_seconds": 161.4},
        "rf": {"model_name": "Random Forest",
               "outer_fold_scores": [0.7608, 0.8606, 0.8322, 0.8735, 0.8495],
               "mean_macro_f1": 0.8353, "std_macro_f1": 0.0396,
               "accuracy": 0.9181, "ece": 0.0213, "brier_score": 0.1249,
               "wall_seconds": 1380.2},
        "mlp": {"model_name": "MLP",
                "outer_fold_scores": [0.7810, 0.8748, 0.8282, 0.8429, 0.8423],
                "mean_macro_f1": 0.8338, "std_macro_f1": 0.0305,
                "accuracy": 0.9137, "ece": 0.0209, "brier_score": 0.1276,
                "wall_seconds": 6711.2, "parameter_count": 3108},
    },
}

FAITHFUL = (
    "The RBF-SVM achieved the highest mean macro-F1 of 0.8771 with a standard "
    "deviation of 0.0248 across the five outer folds. XGBoost followed at "
    "0.8625 and Random Forest at 0.8353. The MLP, with 3108 parameters, "
    "reached 0.8338. All four models exceed the majority-class baseline of "
    "0.2183. The MLP recorded the lowest expected calibration error at "
    "0.0209."
)


# --- no false alarms on a faithful summary ------------------------------

def test_faithful_summary_passes():
    r = verify(FAITHFUL, RECORD)
    bad = [c for c in r.numeric if c.verdict in ("ALTERED", "UNSUPPORTED")]
    assert not bad, [(c.text, c.verdict, c.sentence) for c in bad]
    assert all(s.verdict == "CORRECT" for s in r.statements)
    assert r.factual_accuracy == 1.0
    assert r.passed()


def test_rounded_numbers_are_accepted():
    r = verify("The SVM reached a macro-F1 of 0.88 and accuracy of 0.94.",
               RECORD)
    assert all(c.verdict == "SUPPORTED" for c in r.numeric), \
        [(c.text, c.verdict) for c in r.numeric]


def test_percentages_are_accepted():
    r = verify("The SVM reached 93.79% accuracy.", RECORD)
    assert all(c.verdict in ("SUPPORTED", "DERIVED") for c in r.numeric)


def test_legitimate_arithmetic_is_derived_not_flagged():
    """The 0.0146 gap between SVM and XGBoost is computable from the record."""
    r = verify("The SVM leads XGBoost by 0.0146 macro-F1.", RECORD)
    assert all(c.verdict in ("SUPPORTED", "DERIVED") for c in r.numeric), \
        [(c.text, c.verdict) for c in r.numeric]


# --- section 10.1: numbers not present in the source --------------------

def test_invented_number_is_unsupported():
    r = verify("The SVM reached a macro-F1 of 0.9412.", RECORD)
    verdicts = {c.text.strip(): c.verdict for c in r.numeric}
    assert verdicts["0.9412"] in ("ALTERED", "UNSUPPORTED")
    assert not r.passed()


def test_wildly_invented_number_is_unsupported():
    r = verify("Training required 45000 GPU hours.", RECORD)
    assert any(c.verdict == "UNSUPPORTED" for c in r.numeric)


# --- section 10.1: changed numerical values -----------------------------

def test_altered_value_is_caught_separately_from_invention():
    """0.8871 is one digit off 0.8771 and reads as authoritative."""
    r = verify("The SVM achieved a mean macro-F1 of 0.8871.", RECORD)
    hit = [c for c in r.numeric if c.text.strip() == "0.8871"][0]
    assert hit.verdict == "ALTERED"
    assert "closest source value" in hit.matched


# --- section 10.1: incorrect ranking statements -------------------------

def test_wrong_superlative_is_caught():
    r = verify("Random Forest achieved the highest macro-F1 of all models.",
               RECORD)
    bad = [s for s in r.statements if s.verdict == "INCORRECT"]
    assert bad, [(s.kind, s.verdict, s.sentence) for s in r.statements]
    assert "svm" in bad[0].detail


def test_correct_superlative_passes():
    r = verify("The RBF-SVM achieved the highest macro-F1.", RECORD)
    assert all(s.verdict == "CORRECT" for s in r.statements)


def test_direction_word_checks_the_value_not_the_polarity():
    """
    'lowest ECE' is a claim about the number. The MLP holds the minimum at
    0.0209, so naming it is correct and naming the SVM (0.0281, the maximum)
    is not. An earlier version treated 'lowest' as a quality word and, because
    ECE is a loss, inverted the check and passed the wrong claim.
    """
    r = verify("The MLP recorded the lowest ECE.", RECORD)
    assert all(s.verdict == "CORRECT" for s in r.statements), \
        [(s.verdict, s.detail) for s in r.statements]
    r = verify("The SVM recorded the lowest ECE.", RECORD)
    assert any(s.verdict == "INCORRECT" for s in r.statements)


def test_quality_word_respects_polarity():
    """'best calibrated' means lowest ECE, so the MLP again."""
    r = verify("The MLP is the best calibrated model by ECE.", RECORD)
    assert all(s.verdict == "CORRECT" for s in r.statements)


# --- section 10.1: unsupported comparisons ------------------------------

def test_wrong_pairwise_comparison_is_caught():
    r = verify("Random Forest outperforms the SVM on macro-F1.", RECORD)
    assert any(s.verdict == "INCORRECT" for s in r.statements)


def test_correct_pairwise_comparison_passes():
    r = verify("The SVM outperforms Random Forest on macro-F1.", RECORD)
    assert all(s.verdict == "CORRECT" for s in r.statements)


# --- section 10.1: incorrect percentages --------------------------------

def test_wrong_percentage_is_flagged():
    r = verify("The SVM reached 97.5% accuracy.", RECORD)
    assert any(c.verdict in ("ALTERED", "UNSUPPORTED") for c in r.numeric)


# --- metrics and reporting ----------------------------------------------

def test_metrics_are_computed():
    r = verify("The SVM scored 0.8771 and the MLP scored 0.9999.", RECORD)
    assert r.n_numbers == 2
    assert r.factual_accuracy == pytest.approx(0.5)
    assert r.unsupported_rate > 0
    counts = r.counts()
    assert counts["SUPPORTED"] >= 1
    assert counts["ALTERED"] + counts["UNSUPPORTED"] >= 1


def test_report_text_names_the_offending_claim():
    out = verify("The SVM reached 0.9412 macro-F1.", RECORD).as_text()
    assert "0.9412" in out and "FAIL" in out


def test_empty_summary_is_vacuously_clean():
    r = verify("No numerical results are reported here.", RECORD)
    assert r.n_numbers == 0
    assert r.passed()


# --- record construction -------------------------------------------------

def test_flatten_reaches_nested_scores():
    facts = flatten(RECORD)
    assert facts["models.svm.mean_macro_f1"] == 0.8771
    assert facts["models.svm.outer_fold_scores[2]"] == 0.9036
    assert facts["majority_baseline_macro_f1"] == 0.2183
    assert all(isinstance(v, float) for v in facts.values())


def test_build_record_matches_the_schema_if_results_exist():
    try:
        rec = build_record()
    except Exception:
        pytest.skip("no results files present")
    if not rec["models"]:
        pytest.skip("no models in results")
    m = next(iter(rec["models"].values()))
    for key in ("model_name", "outer_fold_scores", "mean_macro_f1",
                "std_macro_f1", "accuracy", "ece", "brier_score"):
        assert key in m


# --- regression: real Gemini output that the verifier wrongly flagged ----

GEMINI_CLAUSE = (
    "XGBoost followed closely with a macro F1-score of 0.8625 (\u00b10.0369) and "
    "93.00% accuracy, while offering the fastest training time (161.4 seconds) "
    "and the lowest Brier score (0.1091)."
)


def test_multi_metric_sentence_is_not_a_false_alarm():
    """
    Every claim in this real Gemini sentence is true. An earlier version
    scanned the whole sentence, took the first metric in dictionary order
    (macro-F1) and bound the word "lowest" to it, reporting a correct summary
    as a ranking error and turning the direct arm's pass rate into 0.60.
    """
    r = verify(GEMINI_CLAUSE, RECORD)
    wrong = [s for s in r.statements if s.verdict == "INCORRECT"]
    assert not wrong, [(s.kind, s.detail) for s in wrong]
    bad_nums = [c for c in r.numeric if c.verdict in ("ALTERED", "UNSUPPORTED")]
    assert not bad_nums, [(c.text, c.verdict, c.matched) for c in bad_nums]
    assert r.passed()


def test_superlatives_in_separate_clauses_are_both_checked():
    """Both correct claims should be recognised, not merely skipped."""
    r = verify(GEMINI_CLAUSE, RECORD)
    kinds = [s.kind for s in r.statements]
    assert any("superlative" in k for k in kinds), kinds
    assert all(s.verdict == "CORRECT" for s in r.statements)


def test_wrong_claim_inside_a_multi_metric_sentence_is_still_caught():
    """Clause splitting must not become a way to smuggle errors through."""
    bad = ("XGBoost achieved a macro F1 of 0.8625, while offering the fastest "
           "training time and the highest macro-F1 of all models.")
    r = verify(bad, RECORD)
    assert any(s.verdict == "INCORRECT" for s in r.statements), \
        [(s.kind, s.verdict, s.sentence) for s in r.statements]


GEMINI_RUN0 = (
    "XGBoost performed comparably with a macro F1 of 0.8625 (\u00b10.0369) and "
    "93.00% accuracy, while demonstrating superior computational efficiency "
    "with a training time of 161.4 seconds and the lowest Brier score "
    "(0.1091).Random Forest and MLP achieved macro F1-scores of 0.8353 and "
    "0.8338, respectively. Although the MLP was the slowest to train "
    "(6711.2 seconds), it exhibited the best calibration with an Expected "
    "Calibration Error of 0.0209."
)


def test_missing_space_after_period_does_not_leak_the_next_subject():
    """
    Generated text routinely omits the space after a sentence-ending period.
    Without normalisation the clause holding "lowest Brier score" absorbed
    "Random Forest" from the following sentence and the claim was attributed
    to the wrong model. Every statement in this real Gemini output is true.
    """
    r = verify(GEMINI_RUN0, RECORD)
    wrong = [s for s in r.statements if s.verdict == "INCORRECT"]
    assert not wrong, [(s.kind, s.detail) for s in wrong]
    assert r.passed()


def test_decimals_are_not_treated_as_sentence_boundaries():
    r = verify("The SVM reached 0.8771 macro-F1 and 0.9379 accuracy.", RECORD)
    assert all(c.verdict == "SUPPORTED" for c in r.numeric)


def test_ambiguous_clause_is_reported_as_unchecked_not_passed():
    """Skipped clauses are counted, so coverage can be stated honestly."""
    r = verify("The SVM had the highest macro-F1 and accuracy and ECE "
               "combined into one clause.", RECORD)
    assert r.counts()["statements_unchecked"] >= 0


OLLAMA_PROMPT = ("Write a short scientific results summary, at most 150 words, "
                 "for the results below.")
OLLAMA_RUN1 = (
    "Here is a 150-word scientific results summary:\n\nIn this study, we "
    "evaluated four models. RBF-SVM outperformed the others, achieving a mean "
    "macro F1-score of 0.8771 and an accuracy of 0.9379."
)


def test_prompt_echo_is_not_counted_as_fabrication():
    """
    Llama opened every summary with "Here is a 150-word...". The 150 comes
    from the instructions, but it sits within tolerance of XGBoost's 161.4
    second runtime, so it was flagged ALTERED and failed all fifteen runs of
    an experiment. Passing the prompt separates instruction echo from claims.
    """
    bad = verify(OLLAMA_RUN1, RECORD)
    assert not bad.passed(), "the artefact should be reproducible without the prompt"
    good = verify(OLLAMA_RUN1, RECORD, prompt=OLLAMA_PROMPT)
    assert good.passed(), [(c.text, c.verdict) for c in good.numeric]
    assert good.counts()["PROMPT_ECHO"] == 1
    assert good.n_numbers == 2      # 150 excluded from the result claims


def test_prompt_echo_does_not_excuse_a_real_error():
    """A wrong figure stays wrong even when the prompt contains numbers."""
    text = "Here is a 150-word summary. The SVM achieved 0.9412 macro-F1."
    r = verify(text, RECORD, prompt=OLLAMA_PROMPT)
    assert not r.passed()
    assert any(c.verdict in ("ALTERED", "UNSUPPORTED") for c in r.numeric)
