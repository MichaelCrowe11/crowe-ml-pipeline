from typing import Dict, Any

def evaluate_impact(discovery_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate the impact of fungal discoveries on humanity.

    Args:
        discovery_data: A dictionary containing data about the discoveries.

    Returns:
        A dictionary summarizing the impact assessment.
    """
    impact_assessment = {
        'total_discoveries': len(discovery_data.get('breakthrough_compounds', [])),
        'revolutionary_compounds': [],
        'global_health_impact': 0,
        'longevity_applications': [],
        'disease_eradication_candidates': []
    }

    for compound in discovery_data.get('breakthrough_compounds', []):
        impact_score = assess_compound_impact(compound)
        impact_assessment['global_health_impact'] += impact_score

        if impact_score >= 8.0:
            impact_assessment['revolutionary_compounds'].append(compound)

        if compound.get('longevity_potential', False):
            impact_assessment['longevity_applications'].append(compound)

        if compound.get('disease_eradication_potential', False):
            impact_assessment['disease_eradication_candidates'].append(compound)

    return impact_assessment

def assess_compound_impact(compound: Dict[str, Any]) -> int:
    """
    Assess the impact of a single compound.

    Args:
        compound: A dictionary containing data about the compound.

    Returns:
        An integer score representing the impact of the compound.
    """
    score = 0

    if compound.get('therapeutic_relevance', False):
        score += 5

    if compound.get('novel_structure', False):
        score += 3

    if compound.get('high_potency', False):
        score += 2

    if compound.get('low_toxicity', False):
        score += 2

    return score