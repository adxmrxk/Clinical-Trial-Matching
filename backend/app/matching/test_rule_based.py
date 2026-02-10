"""
Test file for Rule-Based Clinical Trial Matcher

Run with: python -m pytest backend/app/matching/test_rule_based.py -v
Or simply: python backend/app/matching/test_rule_based.py
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.matching.rule_based_matcher import (
    RuleBasedMatcher,
    RuleBasedEvaluator,
    evaluate_single_criterion,
    create_matcher,
)


def test_age_criteria():
    """Test age-based criteria evaluation."""
    print("\n" + "="*60)
    print("TEST: Age Criteria")
    print("="*60)

    patient = {"age": 45}

    # Test 1: Age >= 18
    result = evaluate_single_criterion(
        criterion_text="Patients must be 18 years of age or older",
        criterion_type="inclusion",
        patient_profile=patient,
        attribute="age",
        operator=">=",
        value="18"
    )
    print(f"\nCriterion: Age >= 18")
    print(f"Patient age: 45")
    print(f"Status: {result.status}")
    print(f"Explanation: {result.explanation}")
    assert result.status == "satisfied", f"Expected satisfied, got {result.status}"

    # Test 2: Age <= 40 (should fail)
    result = evaluate_single_criterion(
        criterion_text="Patients must be 40 years old or younger",
        criterion_type="inclusion",
        patient_profile=patient,
        attribute="age",
        operator="<=",
        value="40"
    )
    print(f"\nCriterion: Age <= 40")
    print(f"Patient age: 45")
    print(f"Status: {result.status}")
    print(f"Explanation: {result.explanation}")
    assert result.status == "violated", f"Expected violated, got {result.status}"

    print("\n[PASS] Age criteria tests passed!")


def test_sex_criteria():
    """Test sex-based criteria evaluation."""
    print("\n" + "="*60)
    print("TEST: Sex Criteria")
    print("="*60)

    # Test female patient
    patient = {"biological_sex": "female"}

    result = evaluate_single_criterion(
        criterion_text="Female patients only",
        criterion_type="inclusion",
        patient_profile=patient,
        attribute="sex",
        operator="=",
        value="female"
    )
    print(f"\nCriterion: Female patients only")
    print(f"Patient sex: female")
    print(f"Status: {result.status}")
    print(f"Explanation: {result.explanation}")
    assert result.status == "satisfied"

    # Test male patient against female-only
    patient = {"biological_sex": "male"}
    result = evaluate_single_criterion(
        criterion_text="Female patients only",
        criterion_type="inclusion",
        patient_profile=patient,
        attribute="sex",
        operator="=",
        value="female"
    )
    print(f"\nCriterion: Female patients only")
    print(f"Patient sex: male")
    print(f"Status: {result.status}")
    print(f"Explanation: {result.explanation}")
    assert result.status == "violated"

    print("\n[PASS] Sex criteria tests passed!")


def test_ecog_status():
    """Test ECOG performance status criteria."""
    print("\n" + "="*60)
    print("TEST: ECOG Performance Status")
    print("="*60)

    patient = {"ecog_status": 1}

    result = evaluate_single_criterion(
        criterion_text="ECOG performance status 0-2",
        criterion_type="inclusion",
        patient_profile=patient,
        attribute="ecog",
        operator="<=",
        value="2"
    )
    print(f"\nCriterion: ECOG 0-2")
    print(f"Patient ECOG: 1")
    print(f"Status: {result.status}")
    print(f"Explanation: {result.explanation}")
    assert result.status == "satisfied"

    # Test patient with ECOG 3 (should fail)
    patient = {"ecog_status": 3}
    result = evaluate_single_criterion(
        criterion_text="ECOG performance status 0-2",
        criterion_type="inclusion",
        patient_profile=patient,
        attribute="ecog",
        operator="<=",
        value="2"
    )
    print(f"\nCriterion: ECOG 0-2")
    print(f"Patient ECOG: 3")
    print(f"Status: {result.status}")
    print(f"Explanation: {result.explanation}")
    assert result.status == "violated"

    print("\n[PASS] ECOG criteria tests passed!")


def test_lab_values():
    """Test lab value criteria."""
    print("\n" + "="*60)
    print("TEST: Lab Value Criteria")
    print("="*60)

    patient = {
        "lab_values": {
            "hemoglobin": 12.5,
            "creatinine": 1.2,
            "platelet": 150000
        }
    }

    result = evaluate_single_criterion(
        criterion_text="Hemoglobin >= 10 g/dL",
        criterion_type="inclusion",
        patient_profile=patient,
        attribute="hemoglobin",
        operator=">=",
        value="10"
    )
    print(f"\nCriterion: Hemoglobin >= 10 g/dL")
    print(f"Patient hemoglobin: 12.5")
    print(f"Status: {result.status}")
    print(f"Explanation: {result.explanation}")
    assert result.status == "satisfied"

    print("\n[PASS] Lab value criteria tests passed!")


def test_comorbidity_exclusion():
    """Test comorbidity/medical history exclusion criteria."""
    print("\n" + "="*60)
    print("TEST: Comorbidity Exclusion Criteria")
    print("="*60)

    # Patient with heart disease
    patient = {
        "comorbidities": ["coronary artery disease", "hypertension"],
        "primary_condition": "lung cancer"
    }

    result = evaluate_single_criterion(
        criterion_text="No history of heart disease",
        criterion_type="exclusion",
        patient_profile=patient
    )
    print(f"\nCriterion: No history of heart disease (exclusion)")
    print(f"Patient comorbidities: {patient['comorbidities']}")
    print(f"Status: {result.status}")
    print(f"Explanation: {result.explanation}")
    assert result.status == "violated"  # Patient HAS heart disease, so criterion is violated

    # Patient without heart disease
    patient = {
        "comorbidities": ["diabetes", "hypertension"],
        "primary_condition": "lung cancer"
    }
    result = evaluate_single_criterion(
        criterion_text="No history of heart disease",
        criterion_type="exclusion",
        patient_profile=patient
    )
    print(f"\nCriterion: No history of heart disease (exclusion)")
    print(f"Patient comorbidities: {patient['comorbidities']}")
    print(f"Status: {result.status}")
    print(f"Explanation: {result.explanation}")
    assert result.status == "satisfied"  # Patient does NOT have heart disease

    print("\n[PASS] Comorbidity exclusion tests passed!")


def test_lifestyle_criteria():
    """Test lifestyle criteria (smoking, pregnancy)."""
    print("\n" + "="*60)
    print("TEST: Lifestyle Criteria")
    print("="*60)

    # Test smoking exclusion
    patient = {
        "smoking_status": "never",
        "biological_sex": "female",
        "pregnancy_status": "not pregnant"
    }

    result = evaluate_single_criterion(
        criterion_text="Non-smokers only",
        criterion_type="inclusion",
        patient_profile=patient
    )
    print(f"\nCriterion: Non-smokers only")
    print(f"Patient smoking status: never")
    print(f"Status: {result.status}")
    print(f"Explanation: {result.explanation}")
    assert result.status == "satisfied"

    # Test pregnancy exclusion
    result = evaluate_single_criterion(
        criterion_text="Must not be pregnant or lactating",
        criterion_type="exclusion",
        patient_profile=patient
    )
    print(f"\nCriterion: Must not be pregnant or lactating")
    print(f"Patient pregnancy status: not pregnant")
    print(f"Status: {result.status}")
    print(f"Explanation: {result.explanation}")
    assert result.status == "satisfied"

    print("\n[PASS] Lifestyle criteria tests passed!")


def test_full_trial_matching():
    """Test complete trial matching with multiple criteria."""
    print("\n" + "="*60)
    print("TEST: Full Trial Matching")
    print("="*60)

    # Create a comprehensive patient profile
    patient = {
        "age": 55,
        "biological_sex": "male",
        "primary_condition": "non-small cell lung cancer",
        "condition_stage": "Stage IIIB",
        "ecog_status": 1,
        "comorbidities": ["type 2 diabetes", "hypertension"],
        "current_medications": ["metformin", "lisinopril"],
        "prior_treatments": ["chemotherapy"],
        "smoking_status": "former",
        "lab_values": {
            "hemoglobin": 13.2,
            "creatinine": 1.1,
            "platelet": 180000,
            "egfr": 85
        }
    }

    # Define trial criteria
    trial_criteria = [
        {
            "criterion_id": "INC-1",
            "original_text": "Age 18 years or older",
            "criterion_type": "inclusion",
            "attribute": "age",
            "operator": ">=",
            "value": "18"
        },
        {
            "criterion_id": "INC-2",
            "original_text": "Histologically confirmed non-small cell lung cancer",
            "criterion_type": "inclusion",
            "attribute": "diagnosis",
            "operator": "has",
            "value": "non-small cell lung cancer"
        },
        {
            "criterion_id": "INC-3",
            "original_text": "ECOG performance status 0-2",
            "criterion_type": "inclusion",
            "attribute": "ecog",
            "operator": "<=",
            "value": "2"
        },
        {
            "criterion_id": "INC-4",
            "original_text": "Hemoglobin >= 9 g/dL",
            "criterion_type": "inclusion",
            "attribute": "hemoglobin",
            "operator": ">=",
            "value": "9"
        },
        {
            "criterion_id": "EXC-1",
            "original_text": "No history of stroke or TIA",
            "criterion_type": "exclusion",
            "attribute": "comorbidity",
            "operator": "not",
            "value": "stroke"
        },
        {
            "criterion_id": "EXC-2",
            "original_text": "No active HIV infection",
            "criterion_type": "exclusion",
            "attribute": "comorbidity",
            "operator": "not",
            "value": "hiv"
        }
    ]

    # Create matcher and run
    matcher = create_matcher()
    result = matcher.match_patient_to_trial(patient, trial_criteria)

    print(f"\nPatient Profile:")
    print(f"  Age: {patient['age']}")
    print(f"  Condition: {patient['primary_condition']}")
    print(f"  Stage: {patient['condition_stage']}")
    print(f"  ECOG: {patient['ecog_status']}")
    print(f"  Comorbidities: {patient['comorbidities']}")

    print(f"\nTrial has {len(trial_criteria)} criteria")

    print(f"\nResults:")
    print(f"  Overall Status: {result.status.upper()}")
    print(f"  Confidence: {result.confidence_score:.1%}")
    print(f"  Satisfied: {len(result.satisfied_criteria)} criteria")
    print(f"  Violated: {len(result.violated_criteria)} criteria")
    print(f"  Unknown: {len(result.unknown_criteria)} criteria")

    print(f"\n  Explanation: {result.explanation}")

    if result.satisfied_criteria:
        print(f"\n  Satisfied Criteria:")
        for crit in result.satisfied_criteria:
            print(f"    - {crit['criterion_id']}: {crit['original_text'][:50]}...")

    if result.violated_criteria:
        print(f"\n  Violated Criteria:")
        for crit in result.violated_criteria:
            print(f"    - {crit['criterion_id']}: {crit['original_text'][:50]}...")

    if result.unknown_criteria:
        print(f"\n  Unknown Criteria:")
        for crit in result.unknown_criteria:
            print(f"    - {crit['criterion_id']}: {crit['original_text'][:50]}...")

    if result.missing_information:
        print(f"\n  Missing Information: {result.missing_information}")

    print("\n[PASS] Full trial matching test completed!")


def test_missing_information():
    """Test handling of missing patient information."""
    print("\n" + "="*60)
    print("TEST: Missing Information Handling")
    print("="*60)

    # Minimal patient profile
    patient = {
        "age": 35,
        "primary_condition": "breast cancer"
    }

    result = evaluate_single_criterion(
        criterion_text="ECOG performance status 0-1",
        criterion_type="inclusion",
        patient_profile=patient
    )
    print(f"\nCriterion: ECOG 0-1")
    print(f"Patient ECOG: Not provided")
    print(f"Status: {result.status}")
    print(f"Missing attribute: {result.missing_attribute}")
    print(f"Explanation: {result.explanation}")
    assert result.status == "unknown"
    assert result.missing_attribute == "ecog_status"

    print("\n[PASS] Missing information handling test passed!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("RULE-BASED CLINICAL TRIAL MATCHER - TEST SUITE")
    print("="*60)

    try:
        test_age_criteria()
        test_sex_criteria()
        test_ecog_status()
        test_lab_values()
        test_comorbidity_exclusion()
        test_lifestyle_criteria()
        test_missing_information()
        test_full_trial_matching()

        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60 + "\n")

    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
