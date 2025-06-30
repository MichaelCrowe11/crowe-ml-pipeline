import pytest
from src.core.synthesis_predictor import SynthesisPredictor

@pytest.fixture
def synthesis_predictor():
    return SynthesisPredictor()

def test_predict_synthesis_pathway(synthesis_predictor):
    compound_data = {
        'name': 'Test Compound',
        'smiles': 'C1=CC=CC=C1',  # Example SMILES for benzene
        'properties': {
            'molecular_weight': 78.11,
            'logP': 2.13
        }
    }
    result = synthesis_predictor.predict_synthesis_pathway(compound_data)
    assert isinstance(result, dict)
    assert 'pathway' in result
    assert 'yield' in result

def test_invalid_compound_data(synthesis_predictor):
    invalid_data = {
        'name': 'Invalid Compound',
        'smiles': '',  # Invalid SMILES
        'properties': {}
    }
    result = synthesis_predictor.predict_synthesis_pathway(invalid_data)
    assert result is None

def test_edge_case_compound(synthesis_predictor):
    edge_case_data = {
        'name': 'Edge Case Compound',
        'smiles': 'C1CCCC1',  # Example SMILES for cyclohexane
        'properties': {
            'molecular_weight': 84.16,
            'logP': 3.44
        }
    }
    result = synthesis_predictor.predict_synthesis_pathway(edge_case_data)
    assert isinstance(result, dict)
    assert 'pathway' in result
    assert 'yield' in result
    assert result['yield'] >= 0  # Yield should be a non-negative value