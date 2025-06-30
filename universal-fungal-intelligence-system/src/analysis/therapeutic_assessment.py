from typing import Dict, Any

def assess_therapeutic_potential(compound_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess the therapeutic potential of compounds based on their properties and bioactivity.

    Args:
        compound_data: A dictionary containing information about the compounds.

    Returns:
        A dictionary containing the assessment results for each compound.
    """
    assessment_results = {}

    for compound_name, properties in compound_data.items():
        bioactivity_score = properties.get('bioactivity_score', 0)
        toxicity_score = properties.get('toxicity_score', 0)
        therapeutic_index = bioactivity_score / (toxicity_score + 1)  # Avoid division by zero

        assessment_results[compound_name] = {
            'bioactivity_score': bioactivity_score,
            'toxicity_score': toxicity_score,
            'therapeutic_index': therapeutic_index,
            'assessment': 'High Potential' if therapeutic_index > 2 else 'Moderate Potential' if therapeutic_index > 1 else 'Low Potential'
        }

    return assessment_results

def evaluate_compound_for_disease(compound_name: str, disease_targets: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a compound's potential effectiveness against specific disease targets.

    Args:
        compound_name: The name of the compound being evaluated.
        disease_targets: A dictionary containing disease targets and their relevance to the compound.

    Returns:
        A dictionary containing the evaluation results.
    """
    evaluation_results = {
        'compound_name': compound_name,
        'targets_evaluated': {},
        'overall_effectiveness': 'Unknown'
    }

    for target, relevance in disease_targets.items():
        effectiveness = 'Effective' if relevance > 0.5 else 'Ineffective'
        evaluation_results['targets_evaluated'][target] = effectiveness

    if all(effectiveness == 'Effective' for effectiveness in evaluation_results['targets_evaluated'].values()):
        evaluation_results['overall_effectiveness'] = 'Highly Effective'
    elif any(effectiveness == 'Effective' for effectiveness in evaluation_results['targets_evaluated'].values()):
        evaluation_results['overall_effectiveness'] = 'Partially Effective'

    return evaluation_results

def summarize_therapeutic_assessment(assessment_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize the therapeutic assessment results for reporting.

    Args:
        assessment_results: A dictionary containing assessment results for each compound.

    Returns:
        A summary dictionary with key insights.
    """
    summary = {
        'total_compounds_assessed': len(assessment_results),
        'high_potential_compounds': [],
        'moderate_potential_compounds': [],
        'low_potential_compounds': []
    }

    for compound_name, result in assessment_results.items():
        if result['assessment'] == 'High Potential':
            summary['high_potential_compounds'].append(compound_name)
        elif result['assessment'] == 'Moderate Potential':
            summary['moderate_potential_compounds'].append(compound_name)
        else:
            summary['low_potential_compounds'].append(compound_name)

    return summary