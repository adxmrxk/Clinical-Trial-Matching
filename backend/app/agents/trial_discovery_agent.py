from typing import Any, Dict, List
from .base_agent import BaseAgent
from ..schemas.patient import PatientProfile
from ..schemas.trial import ClinicalTrial, TrialSearchParams
from ..services.clinical_trials_api import clinical_trials_service


class TrialDiscoveryAgent(BaseAgent):
    """
    Agent responsible for discovering relevant clinical trials
    based on patient profile.
    """

    def __init__(self):
        super().__init__(
            name="Trial Discovery Agent",
            description="Search and filter clinical trials relevant to the patient's condition."
        )

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Discover clinical trials based on patient profile.

        Input:
            - patient_profile: The patient's profile
            - max_results: Maximum number of trials to return

        Output:
            - trials: List of relevant clinical trials
            - search_params: The search parameters used
        """
        profile: PatientProfile = input_data.get("patient_profile", PatientProfile())
        max_results = input_data.get("max_results", 20)

        # Build search parameters from patient profile
        search_params = self._build_search_params(profile, max_results)

        # Query the clinical trials API
        trials = await clinical_trials_service.search_trials(search_params)

        # Filter and rank trials
        ranked_trials = self._rank_trials(trials, profile)

        return {
            "trials": ranked_trials,
            "search_params": search_params,
            "total_found": len(trials)
        }

    def _build_search_params(self, profile: PatientProfile, max_results: int) -> TrialSearchParams:
        """Build search parameters from patient profile."""
        params = TrialSearchParams(max_results=max_results)

        # Use primary condition for search
        if profile.primary_condition:
            params.condition = profile.primary_condition

        # Build location filter
        location_parts = []
        if profile.city:
            location_parts.append(profile.city)
        if profile.state_province:
            location_parts.append(profile.state_province)
        if profile.country:
            location_parts.append(profile.country)

        if location_parts:
            params.location = ", ".join(location_parts)

        # Only recruiting trials
        params.status = ["RECRUITING"]

        return params

    def _rank_trials(self, trials: List[ClinicalTrial], profile: PatientProfile) -> List[ClinicalTrial]:
        """
        Rank trials by relevance to patient profile.
        This is a simple ranking - can be enhanced with LLM-based relevance scoring.
        """

        def score_trial(trial: ClinicalTrial) -> float:
            score = 0.0

            # Condition match
            if profile.primary_condition and isinstance(profile.primary_condition, str):
                condition_lower = profile.primary_condition.lower()
                for trial_condition in trial.conditions:
                    if trial_condition and condition_lower in trial_condition.lower():
                        score += 10.0
                        break

            # Location match
            if profile.country and isinstance(profile.country, str):
                for loc in trial.locations:
                    loc_country = loc.get("country", "") or ""
                    if loc_country.lower() == profile.country.lower():
                        score += 5.0
                        break

            if profile.state_province and isinstance(profile.state_province, str):
                for loc in trial.locations:
                    loc_state = loc.get("state", "") or ""
                    if loc_state.lower() == profile.state_province.lower():
                        score += 3.0
                        break

            # Age match
            if profile.age and trial.minimum_age and trial.maximum_age:
                try:
                    min_age = self._parse_age(trial.minimum_age)
                    max_age = self._parse_age(trial.maximum_age)
                    if min_age <= profile.age <= max_age:
                        score += 5.0
                except:
                    pass

            # Sex match
            if profile.biological_sex and trial.sex:
                trial_sex = trial.sex.lower() if isinstance(trial.sex, str) else ""
                profile_sex = profile.biological_sex.value.lower() if hasattr(profile.biological_sex, 'value') else str(profile.biological_sex).lower()
                if trial_sex == "all" or trial_sex == profile_sex:
                    score += 2.0

            return score

        # Sort by score descending
        return sorted(trials, key=score_trial, reverse=True)

    def _parse_age(self, age_str: str) -> int:
        """Parse age string like '18 Years' to integer."""
        if not age_str:
            return 0
        parts = age_str.split()
        if parts:
            try:
                return int(parts[0])
            except ValueError:
                return 0
        return 0


# Singleton instance
trial_discovery_agent = TrialDiscoveryAgent()
