"""
Clinical Trial Matching Module

This module provides rule-based matching capabilities for
evaluating patient eligibility against clinical trial criteria.
"""

from .rule_based_matcher import (
    # Main classes
    RuleBasedMatcher,
    RuleBasedEvaluator,

    # Data classes
    EvaluationResult,
    TrialEligibilityResult,
    CriterionCategory,
    MedicalDictionary,
    NumericPatterns,

    # Convenience functions
    create_matcher,
    evaluate_single_criterion,

    # Global instances
    MEDICAL_DICTIONARY,
    NUMERIC_PATTERNS,
)

__all__ = [
    "RuleBasedMatcher",
    "RuleBasedEvaluator",
    "EvaluationResult",
    "TrialEligibilityResult",
    "CriterionCategory",
    "MedicalDictionary",
    "NumericPatterns",
    "create_matcher",
    "evaluate_single_criterion",
    "MEDICAL_DICTIONARY",
    "NUMERIC_PATTERNS",
]
