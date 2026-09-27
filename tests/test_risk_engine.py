"""Risk engine: deterministic levels, heuristic score, advisory evidence ignored."""

import pytest

from app import risk_engine
from app.models import Evidence, Severity


def ev(rule, advisory=False):
    return Evidence(rule=rule, category="metadata", title=rule, detail="", advisory=advisory)


def assess(rules, basis=True, advisory=()):
    items = [ev(r) for r in rules] + [ev(r, advisory=True) for r in advisory]
    risk_engine.apply_rules(items)
    return risk_engine.assess(items, has_verifiable_basis=basis)


def test_no_indicators_with_basis():
    r = assess([])
    assert r.level == "NO_SIGNIFICANT_INDICATORS" and r.score == 0


def test_no_indicators_without_basis_is_insufficient_evidence():
    assert assess([], basis=False).level == "INSUFFICIENT_EVIDENCE"


def test_single_high_indicator_is_high_priority():
    r = assess(["CAPTURE_BEFORE_ORDER"])
    assert r.level == "HIGH_REVIEW_PRIORITY" and r.score == 50


def test_single_medium_indicator_is_review():
    assert assess(["EDITING_SOFTWARE"]).level == "REVIEW_RECOMMENDED"


def test_medium_indicators_can_add_up_to_high():
    r = assess(["THUMBNAIL_MISMATCH", "EDITING_SOFTWARE"])
    assert r.score == 50 and r.level == "HIGH_REVIEW_PRIORITY"


def test_weak_indicators_alone_do_not_trigger_review():
    r = assess(["FORMAT_MISMATCH", "AI_TYPICAL_DIMENSIONS"])
    assert r.score == 10 and r.level == "NO_SIGNIFICANT_INDICATORS"


def test_weak_indicators_without_basis_stay_insufficient():
    assert assess(["FORMAT_MISMATCH"], basis=False).level == "INSUFFICIENT_EVIDENCE"


def test_score_is_capped_at_100():
    r = assess(["AI_PROVENANCE_DECLARED", "DUPLICATE_EXACT_OTHER_ORDER", "CAPTURE_BEFORE_ORDER"])
    assert r.score == 100


def test_advisory_evidence_never_scores_even_with_known_rule():
    r = assess([], advisory=["AI_PROVENANCE_DECLARED", "THUMBNAIL_MISMATCH"])
    assert r.score == 0 and r.level == "NO_SIGNIFICANT_INDICATORS"


def test_info_rules_and_unknown_rules_score_zero():
    items = [ev("INFO_NO_METADATA"), ev("SOMETHING_NEW")]
    risk_engine.apply_rules(items)
    assert all(i.points == 0 for i in items)


def test_reasons_are_ordered_by_points():
    r = assess(["EDITING_SOFTWARE", "AI_PROVENANCE_DECLARED"])
    assert r.reasons[0].startswith("AI_PROVENANCE_DECLARED")


def test_score_note_disclaims_probability():
    assert "not a probability" in assess([]).score_note


@pytest.mark.parametrize("level", ["HIGH_REVIEW_PRIORITY", "REVIEW_RECOMMENDED", "NO_SIGNIFICANT_INDICATORS", "INSUFFICIENT_EVIDENCE"])
def test_no_recommendation_approves_or_rejects(level):
    text = risk_engine._RECOMMENDATIONS[level].lower()
    assert "approve" not in text and "reject" not in text


def test_every_rule_has_consistent_severity_and_points():
    for name, rule in risk_engine.RULES.items():
        assert rule.points > 0, name
        assert rule.severity != Severity.INFO, name
