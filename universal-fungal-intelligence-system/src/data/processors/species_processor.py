from typing import Dict, Any

def process_species_data(species_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process species data collected from various sources.
    
    Args:
        species_data: A dictionary containing species data.
        
    Returns:
        A dictionary with processed species data.
    """
    processed_data = {}
    
    for species_id, data in species_data.items():
        processed_data[species_id] = {
            'scientific_name': data.get('scientific_name'),
            'genus': data.get('genus'),
            'species': data.get('species'),
            'phylum': data.get('phylum'),
            'habitat': data.get('habitat'),
            'economic_importance': data.get('economic_importance'),
            'documented_compounds': data.get('documented_compounds'),
            'bioactivity_reports': data.get('bioactivity_reports')
        }
    
    return processed_data

def filter_species_by_habitat(species_data: Dict[str, Any], habitat: str) -> Dict[str, Any]:
    """
    Filter species data by habitat.
    
    Args:
        species_data: A dictionary containing species data.
        habitat: The habitat to filter by.
        
    Returns:
        A dictionary with species data filtered by the specified habitat.
    """
    filtered_species = {
        species_id: data for species_id, data in species_data.items() if data.get('habitat') == habitat
    }
    
    return filtered_species

def summarize_species_data(species_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize species data to provide insights.
    
    Args:
        species_data: A dictionary containing species data.
        
    Returns:
        A summary dictionary with insights about the species data.
    """
    summary = {
        'total_species': len(species_data),
        'unique_phyla': len(set(data.get('phylum') for data in species_data.values())),
        'economic_importance_count': sum(1 for data in species_data.values() if data.get('economic_importance')),
        'documented_compounds_count': sum(len(data.get('documented_compounds', [])) for data in species_data.values())
    }
    
    return summary