from datetime import datetime

def generate_impact_report(analysis_results):
    report = {
        'report_metadata': {
            'report_id': f"IMPACT_REPORT_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            'generation_datetime': datetime.utcnow().isoformat(),
            'analysis_summary': analysis_results.get('summary', ''),
            'breakthrough_compounds_count': len(analysis_results.get('breakthrough_compounds', {})),
            'therapeutic_applications_count': len(analysis_results.get('therapeutic_applications', {})),
        },
        'impact_assessment': {
            'global_health_impact': analysis_results.get('global_health_impact', {}),
            'longevity_applications': analysis_results.get('longevity_applications', {}),
            'disease_eradication_candidates': analysis_results.get('disease_eradication_candidates', {}),
        },
        'recommendations': generate_recommendations(analysis_results)
    }
    
    return report

def generate_recommendations(analysis_results):
    recommendations = []
    
    if analysis_results.get('global_health_impact'):
        recommendations.append("Prioritize compounds with high global health impact.")
    
    if analysis_results.get('longevity_applications'):
        recommendations.append("Explore further research on longevity enhancement compounds.")
    
    if analysis_results.get('disease_eradication_candidates'):
        recommendations.append("Focus on developing therapies for identified disease eradication candidates.")
    
    return recommendations