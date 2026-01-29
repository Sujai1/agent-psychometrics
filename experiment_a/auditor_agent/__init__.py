"""Auditor agent for SWE-bench task difficulty assessment.

This module provides an LLM-based agent that explores SWE-bench task
environments and rates them on difficulty-related axes.
"""

from experiment_a.auditor_agent.prompts import (
    AUDITOR_FEATURES,
    AuditorFeature,
    build_auditor_system_prompt,
    get_feature_names,
    VERIFICATION_PROMPT,
)

__all__ = [
    "AUDITOR_FEATURES",
    "AuditorFeature",
    "build_auditor_system_prompt",
    "get_feature_names",
    "VERIFICATION_PROMPT",
]