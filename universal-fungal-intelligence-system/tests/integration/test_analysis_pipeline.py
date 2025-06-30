import pytest
from src.core.fungal_intelligence import UniversalFungalIntelligence

@pytest.fixture
def fungal_intelligence_system():
    """Fixture to initialize the Universal Fungal Intelligence System."""
    return UniversalFungalIntelligence()

def test_global_fungal_analysis(fungal_intelligence_system):
    """Test the global fungal kingdom analysis process."""
    analysis_results = fungal_intelligence_system.analyze_global_fungal_kingdom()
    
    assert isinstance(analysis_results, dict)
    assert 'analysis_metadata' in analysis_results
    assert 'species_analysis' in analysis_results
    assert 'chemical_discoveries' in analysis_results
    assert 'breakthrough_modifications' in analysis_results
    assert 'therapeutic_potential' in analysis_results
    assert 'synthesis_pathways' in analysis_results
    assert 'humanity_impact_assessment' in analysis_results
    assert 'priority_research_targets' in analysis_results

def test_collect_all_fungal_species(fungal_intelligence_system):
    """Test the collection of all documented fungal species."""
    species_data = fungal_intelligence_system._collect_all_fungal_species()
    
    assert isinstance(species_data, dict)
    assert 'collection_metadata' in species_data
    assert 'taxonomic_data' in species_data
    assert 'known_compounds' in species_data
    assert species_data['collection_metadata']['total_species_documented'] > 0

def test_analyze_species_chemistry(fungal_intelligence_system):
    """Test the analysis of chemical composition of fungal species."""
    species_data = fungal_intelligence_system._collect_all_fungal_species()
    chemistry_analysis = fungal_intelligence_system._analyze_species_chemistry(species_data)
    
    assert isinstance(chemistry_analysis, dict)
    assert 'compound_profiles' in chemistry_analysis
    assert 'biosynthetic_pathways' in chemistry_analysis
    assert 'chemical_diversity' in chemistry_analysis

def test_identify_breakthrough_compounds(fungal_intelligence_system):
    """Test the identification of breakthrough compounds."""
    species_data = fungal_intelligence_system._collect_all_fungal_species()
    chemistry_analysis = fungal_intelligence_system._analyze_species_chemistry(species_data)
    breakthrough_compounds = fungal_intelligence_system._identify_breakthrough_compounds(chemistry_analysis)
    
    assert isinstance(breakthrough_compounds, dict)
    assert 'high_priority_compounds' in breakthrough_compounds
    assert len(breakthrough_compounds['high_priority_compounds']) > 0

def test_predict_molecular_modifications(fungal_intelligence_system):
    """Test the prediction of molecular modifications."""
    species_data = fungal_intelligence_system._collect_all_fungal_species()
    chemistry_analysis = fungal_intelligence_system._analyze_species_chemistry(species_data)
    breakthrough_compounds = fungal_intelligence_system._identify_breakthrough_compounds(chemistry_analysis)
    molecular_modifications = fungal_intelligence_system._predict_molecular_modifications(breakthrough_compounds)
    
    assert isinstance(molecular_modifications, dict)
    assert 'enhancement_predictions' in molecular_modifications
    assert len(molecular_modifications['enhancement_predictions']) > 0

def test_assess_therapeutic_potential(fungal_intelligence_system):
    """Test the assessment of therapeutic potential."""
    species_data = fungal_intelligence_system._collect_all_fungal_species()
    chemistry_analysis = fungal_intelligence_system._analyze_species_chemistry(species_data)
    breakthrough_compounds = fungal_intelligence_system._identify_breakthrough_compounds(chemistry_analysis)
    molecular_modifications = fungal_intelligence_system._predict_molecular_modifications(breakthrough_compounds)
    therapeutic_assessment = fungal_intelligence_system._assess_therapeutic_potential(molecular_modifications)
    
    assert isinstance(therapeutic_assessment, dict)
    assert len(therapeutic_assessment) > 0

def test_evaluate_humanity_impact(fungal_intelligence_system):
    """Test the evaluation of humanity impact."""
    species_data = fungal_intelligence_system._collect_all_fungal_species()
    chemistry_analysis = fungal_intelligence_system._analyze_species_chemistry(species_data)
    breakthrough_compounds = fungal_intelligence_system._identify_breakthrough_compounds(chemistry_analysis)
    molecular_modifications = fungal_intelligence_system._predict_molecular_modifications(breakthrough_compounds)
    therapeutic_assessment = fungal_intelligence_system._assess_therapeutic_potential(molecular_modifications)
    humanity_impact = fungal_intelligence_system._evaluate_humanity_impact(therapeutic_assessment)
    
    assert isinstance(humanity_impact, dict)
    assert 'revolutionary_compounds' in humanity_impact
    assert len(humanity_impact['revolutionary_compounds']) > 0