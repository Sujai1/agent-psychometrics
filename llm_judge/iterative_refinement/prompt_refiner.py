"""LLM-based prompt refinement using residual analysis.

Uses an LLM to propose improved feature definitions based on:
1. Current feature correlations and entropies
2. High-residual tasks where predictions fail
3. Feature coefficients showing which features matter most
4. Coefficient direction validation (does the sign match semantic expectation?)
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from llm_judge.iterative_refinement.prompt_store import (
    FeatureDefinition,
    generate_prompt_from_schema,
)
from llm_judge.iterative_refinement.residual_analyzer import (
    ResidualAnalysis,
    format_residual_analysis_for_llm,
)


def get_expected_directions(
    features: List[FeatureDefinition],
    model: str = "gpt-5.2",
) -> Dict[str, Dict[str, str]]:
    """Use LLM to determine expected coefficient direction for each feature.

    Args:
        features: List of feature definitions
        model: Model to use for inference

    Returns:
        Dict mapping feature_name -> {"direction": "HARDER"|"EASIER", "reason": str}
    """
    features_text = []
    for f in features:
        features_text.append(f"""
Feature: {f.name}
Description: {f.description}
Scale: {f.min_value} ({f.scale_low}) to {f.max_value} ({f.scale_high})
""")

    prompt = """You are analyzing features used to predict task difficulty for AI coding agents.

Higher difficulty (β) = harder task for AI agents to solve.

For each feature below, determine whether HIGHER values of that feature should make tasks:
- HARDER (positive correlation with difficulty)
- EASIER (negative correlation with difficulty)

Think about what the feature measures and how it relates to AI agent performance.

Features:
""" + "\n".join(features_text) + """

Output a JSON object mapping feature_name -> {"direction": "HARDER" or "EASIER", "reason": "brief explanation"}
"""

    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        max_output_tokens=2000,
    )

    text = response.output_text.strip()
    if "```json" in text:
        json_str = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        json_str = text.split("```")[1].split("```")[0].strip()
    else:
        json_str = text

    return json.loads(json_str)


def validate_coefficient_directions(
    features: List[FeatureDefinition],
    coefficients: Dict[str, float],
    expected_directions: Dict[str, Dict[str, str]],
    min_coef_magnitude: float = 0.1,
) -> List[Dict[str, Any]]:
    """Identify features where coefficient sign doesn't match expected direction.

    Args:
        features: List of feature definitions
        coefficients: Dict mapping feature_name -> coefficient value
        expected_directions: Dict from get_expected_directions()
        min_coef_magnitude: Minimum |coefficient| to flag as mismatch

    Returns:
        List of mismatched features with details
    """
    mismatches = []
    for f in features:
        coef = coefficients.get(f.name, 0)
        actual_direction = "HARDER" if coef > 0 else "EASIER"

        expected_info = expected_directions.get(f.name, {})
        expected_direction = expected_info.get("direction", "UNKNOWN")
        reason = expected_info.get("reason", "")

        if expected_direction != actual_direction and abs(coef) >= min_coef_magnitude:
            mismatches.append({
                "name": f.name,
                "coefficient": coef,
                "expected_direction": expected_direction,
                "actual_direction": actual_direction,
                "reason": reason,
                "severity": "HIGH" if abs(coef) > 0.5 else "MEDIUM",
            })

    return mismatches


@dataclass
class RefinementProposal:
    """A proposed refinement to the feature schema."""

    # New feature definitions
    new_features: List[FeatureDefinition]

    # Changes summary
    features_added: List[str]
    features_removed: List[str]
    features_modified: List[str]

    # Reasoning from LLM
    reasoning: str

    # Archived features (removed from previous version)
    archived_features: List[FeatureDefinition]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "features_added": self.features_added,
            "features_removed": self.features_removed,
            "features_modified": self.features_modified,
            "reasoning": self.reasoning,
            "n_new_features": len(self.new_features),
        }


REFINEMENT_SYSTEM_PROMPT = """You are an expert at designing feature extraction prompts for difficulty prediction.

Your task is to refine a set of features used by an LLM judge to predict how difficult programming tasks are for AI coding agents.

The current features are used to predict IRT difficulty scores (β). Higher β = harder task.

CRITICAL CONSTRAINT: You MUST make INCREMENTAL changes only.
- You MUST KEEP features marked as "PROTECTED" (high coefficient or entropy) unless explicitly flagged as problematic
- You may MODIFY at most 2-3 features per iteration (refine scale anchors, improve clarity)
- You may REMOVE at most 1-2 features that are clearly redundant or uninformative
- You may ADD at most 1-2 new features to address specific failure cases

Based on the prediction failures shown, propose improvements:
1. KEEP features with high regression coefficients - they are predictive
2. MODIFY features that have poor scale anchors or unclear definitions
3. REMOVE only features that are redundant (r > 0.9 with another) or have very low entropy
4. ADD new features that would distinguish the failure cases shown
5. FIX features with WRONG DIRECTION coefficients - if a feature that should make tasks HARDER has a negative coefficient (or vice versa), the scale may be inverted or the definition unclear

Guidelines:
- Keep total features between 6-12 (balance between coverage and noise)
- Each feature should have clear, distinct scale anchors
- Features should be observable from the problem statement and gold patch
- Prefer modifying existing features over replacing them entirely
- Consider what makes tasks "deceptively" hard or easy

Output a JSON object with:
1. "features": array of feature definitions (include ALL features, both kept and modified)
2. "changes_summary": brief description of what changed and why
3. "reasoning": explanation of why these changes should improve predictions"""


def build_refinement_prompt(
    current_features: List[FeatureDefinition],
    analysis: ResidualAnalysis,
    quick_eval_metrics: Dict[str, Any],
    direction_mismatches: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build the prompt for the LLM refiner.

    Args:
        current_features: Current feature definitions
        analysis: Residual analysis with high-residual tasks
        quick_eval_metrics: Metrics from quick evaluation (entropy, correlations)
        direction_mismatches: Features where coefficient sign doesn't match expected direction

    Returns:
        Prompt string for refinement
    """
    # Identify if there's an outstanding feature (coefficient > 2x the next best)
    coefficients = analysis.feature_coefficients
    sorted_by_coef = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
    protected_features = set()
    if len(sorted_by_coef) >= 2:
        top_coef = abs(sorted_by_coef[0][1])
        second_coef = abs(sorted_by_coef[1][1])
        # Only protect if clearly outstanding (2x better than second)
        if top_coef > 2 * second_coef and top_coef > 0.1:
            protected_features.add(sorted_by_coef[0][0])

    # Also get entropy info
    entropies = quick_eval_metrics.get("feature_entropies", {})

    lines = [
        "# Current Feature Schema",
        "",
        f"The current schema has {len(current_features)} features.",
        f"Correlation with ground truth difficulty: r = {quick_eval_metrics.get('pearson_r', 'N/A')}",
        "",
        "## Feature Definitions",
        "",
    ]

    for f in current_features:
        # Mark protected features (only truly outstanding ones)
        protection_status = ""
        if f.name in protected_features:
            coef = coefficients.get(f.name, 0)
            protection_status = f" 🔒 PROTECTED (coef={coef:+.3f}, clearly best predictor)"

        entropy = entropies.get(f.name, 0)
        entropy_note = f" [entropy={entropy:.2f}]" if entropy > 0 else ""

        lines.append(f"### {f.name} ({f.min_value}-{f.max_value}){protection_status}{entropy_note}")
        lines.append(f"*{f.description}*")
        lines.append(f"- Low ({f.min_value}): {f.scale_low}")
        lines.append(f"- High ({f.max_value}): {f.scale_high}")
        lines.append("")

    # Add entropy information with clearer thresholds
    if entropies:
        lines.extend(
            [
                "## Feature Quality Metrics",
                "",
                "### Entropy (normalized 0-1, higher = uses more of scale)",
                "",
            ]
        )
        for name, entropy in sorted(entropies.items(), key=lambda x: x[1]):
            flag = " ⚠️ VERY LOW - consider modifying scale" if entropy < 0.5 else ""
            lines.append(f"- {name}: {entropy:.2f}{flag}")

    # Add redundancy information
    redundant = quick_eval_metrics.get("redundant_pairs", [])
    if redundant:
        lines.extend(
            [
                "",
                "### Redundant Feature Pairs (r > 0.9)",
                "",
            ]
        )
        for pair in redundant:
            if isinstance(pair, dict):
                lines.append(f"- {pair['f1']} ↔ {pair['f2']}: r = {pair['r']:.2f}")
            else:
                lines.append(f"- {pair[0]} ↔ {pair[1]}: r = {pair[2]:.2f}")

    # Add direction mismatch information
    if direction_mismatches:
        lines.extend(
            [
                "",
                "### ⚠️ COEFFICIENT DIRECTION MISMATCHES",
                "",
                "These features have coefficients that go the WRONG direction semantically.",
                "A feature that should make tasks HARDER has a negative coefficient (or vice versa).",
                "This suggests the feature definition is unclear or the scale is inverted.",
                "PRIORITY: Fix or remove these features.",
                "",
            ]
        )
        for m in direction_mismatches:
            lines.append(f"**{m['name']}** (coef={m['coefficient']:+.3f}, severity={m['severity']})")
            lines.append(f"  - Expected: Higher values → {m['expected_direction']} ({m['reason']})")
            lines.append(f"  - Actual: Higher values → {m['actual_direction']}")
            lines.append("")

    # Add residual analysis
    lines.append("")
    lines.append(format_residual_analysis_for_llm(analysis, current_features))

    lines.extend(
        [
            "",
            "# Your Task",
            "",
            "Propose INCREMENTAL improvements to the feature schema.",
            "",
            "CONSTRAINTS:",
            "- KEEP most existing features (especially those with high coefficients)",
            "- MODIFY at most 2-3 features (refine their scales or descriptions)",
            "- REMOVE at most 1-2 clearly problematic features (very low entropy or highly redundant)",
            "- ADD at most 1-2 new features to address the failure cases",
            "",
            "DO NOT replace all features - make targeted improvements only.",
            "",
            "Output JSON in this format:",
            "```json",
            "{",
            '  "features": [',
            "    {",
            '      "name": "feature_name",',
            '      "description": "What this feature measures",',
            '      "scale_low": "Description of lowest value",',
            '      "scale_high": "Description of highest value",',
            '      "min_value": 0,',
            '      "max_value": 5,',
            '      "extraction_prompt": "Full prompt text for extracting this feature"',
            "    },",
            "    ...",
            "  ],",
            '  "changes_summary": "Brief description of changes",',
            '  "reasoning": "Why these changes should improve predictions"',
            "}",
            "```",
        ]
    )

    return "\n".join(lines)


def propose_refinement(
    current_features: List[FeatureDefinition],
    analysis: ResidualAnalysis,
    quick_eval_metrics: Dict[str, Any],
    model: str = "gpt-5.2",
    validate_directions: bool = True,
) -> RefinementProposal:
    """Use LLM to propose refined feature definitions.

    Args:
        current_features: Current feature definitions
        analysis: Residual analysis with failure cases
        quick_eval_metrics: Metrics from quick evaluation
        model: Model to use for refinement
        validate_directions: Whether to check coefficient direction matches

    Returns:
        RefinementProposal with new feature schema
    """
    client = OpenAI()

    # Optionally validate coefficient directions
    direction_mismatches = None
    if validate_directions:
        try:
            expected_directions = get_expected_directions(current_features, model)
            direction_mismatches = validate_coefficient_directions(
                current_features,
                analysis.feature_coefficients,
                expected_directions,
            )
            if direction_mismatches:
                print(f"   Found {len(direction_mismatches)} coefficient direction mismatches")
                for m in direction_mismatches:
                    print(f"     - {m['name']}: expected {m['expected_direction']}, got {m['actual_direction']}")
        except Exception as e:
            print(f"   Warning: Could not validate directions: {e}")

    prompt = build_refinement_prompt(
        current_features, analysis, quick_eval_metrics, direction_mismatches
    )

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_output_tokens=4000,
    )

    # Parse response
    text = response.output_text.strip()

    # Extract JSON from response
    if "```json" in text:
        json_str = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        json_str = text.split("```")[1].split("```")[0].strip()
    else:
        json_str = text

    data = json.loads(json_str)

    # Parse features
    new_features = []
    for f in data.get("features", []):
        new_features.append(
            FeatureDefinition(
                name=f["name"],
                description=f["description"],
                scale_low=f["scale_low"],
                scale_high=f["scale_high"],
                min_value=f["min_value"],
                max_value=f["max_value"],
                extraction_prompt=f["extraction_prompt"],
            )
        )

    # Determine changes
    current_names = {f.name for f in current_features}
    new_names = {f.name for f in new_features}

    features_added = list(new_names - current_names)
    features_removed = list(current_names - new_names)
    features_modified = [
        name
        for name in current_names & new_names
        if _feature_modified(
            next(f for f in current_features if f.name == name),
            next(f for f in new_features if f.name == name),
        )
    ]

    # Archive removed features
    archived = [f for f in current_features if f.name in features_removed]

    return RefinementProposal(
        new_features=new_features,
        features_added=features_added,
        features_removed=features_removed,
        features_modified=features_modified,
        reasoning=data.get("reasoning", "") + "\n\n" + data.get("changes_summary", ""),
        archived_features=archived,
    )


def _feature_modified(old: FeatureDefinition, new: FeatureDefinition) -> bool:
    """Check if a feature was meaningfully modified."""
    return (
        old.description != new.description
        or old.scale_low != new.scale_low
        or old.scale_high != new.scale_high
        or old.min_value != new.min_value
        or old.max_value != new.max_value
    )


def apply_refinement_constraints(
    proposal: RefinementProposal,
    current_features: Optional[List[FeatureDefinition]] = None,
    min_features: int = 6,
    max_features: int = 12,
    max_removals: int = 2,
    max_additions: int = 2,
) -> RefinementProposal:
    """Apply constraints to ensure valid and incremental feature schema.

    Args:
        proposal: The raw proposal from LLM
        current_features: Current feature definitions (for enforcing incremental changes)
        min_features: Minimum allowed features
        max_features: Maximum allowed features
        max_removals: Maximum features that can be removed per iteration
        max_additions: Maximum features that can be added per iteration

    Returns:
        Constrained proposal
    """
    features = proposal.new_features
    features_added = proposal.features_added
    features_removed = proposal.features_removed
    archived_features = proposal.archived_features

    # If too many features were removed and we have the originals, add some back
    if current_features and len(features_removed) > max_removals:
        print(f"Warning: {len(features_removed)} features removed, limiting to {max_removals}")

        # Keep only the first max_removals removals
        allowed_removals = set(features_removed[:max_removals])
        features_to_restore = [
            f for f in current_features
            if f.name in features_removed and f.name not in allowed_removals
        ]

        # Add back the features that shouldn't have been removed
        new_names = {f.name for f in features}
        for f in features_to_restore:
            if f.name not in new_names:
                features.append(f)
                print(f"  Restored feature: {f.name}")

        # Update the removed list
        features_removed = list(allowed_removals)
        archived_features = [f for f in archived_features if f.name in allowed_removals]

    # If too many features were added, trim them
    if len(features_added) > max_additions:
        print(f"Warning: {len(features_added)} features added, limiting to {max_additions}")
        allowed_additions = set(features_added[:max_additions])
        features = [f for f in features if f.name not in features_added or f.name in allowed_additions]
        features_added = list(allowed_additions)

    # Ensure we have at least min_features
    if len(features) < min_features:
        print(f"Warning: Only {len(features)} features proposed, keeping all")

    # Trim to max_features if needed
    if len(features) > max_features:
        print(f"Warning: {len(features)} features proposed, trimming to {max_features}")
        features = features[:max_features]

    return RefinementProposal(
        new_features=features,
        features_added=features_added,
        features_removed=features_removed,
        features_modified=proposal.features_modified,
        reasoning=proposal.reasoning,
        archived_features=archived_features,
    )