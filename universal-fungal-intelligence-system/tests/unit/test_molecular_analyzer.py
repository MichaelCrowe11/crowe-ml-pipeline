import pytest
from src.core.molecular_analyzer import MolecularAnalyzer

@pytest.fixture
def molecular_analyzer():
    return MolecularAnalyzer()

def test_analyze_structure_valid(molecular_analyzer):
    smiles = "CCO"  # Ethanol
    result = molecular_analyzer.analyze_structure(smiles)
    assert result is not None
    assert 'molecular_weight' in result
    assert 'logP' in result

def test_analyze_structure_invalid(molecular_analyzer):
    smiles = "INVALID_SMILES"
    result = molecular_analyzer.analyze_structure(smiles)
    assert result is None

def test_identify_compounds(molecular_analyzer):
    compounds = molecular_analyzer.identify_compounds("CCO")
    assert len(compounds) > 0
    assert "ethanol" in [compound['name'].lower() for compound in compounds]

def test_calculate_properties(molecular_analyzer):
    smiles = "CCO"
    properties = molecular_analyzer.calculate_properties(smiles)
    assert properties['molecular_weight'] > 0
    assert properties['logP'] is not None

def test_handle_empty_smiles(molecular_analyzer):
    result = molecular_analyzer.analyze_structure("")
    assert result is None