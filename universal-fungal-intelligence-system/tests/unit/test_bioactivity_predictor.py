import pytest
from src.core.bioactivity_predictor import BioactivityPredictor

@pytest.fixture
def bioactivity_predictor():
    return BioactivityPredictor()

def test_predict_bioactivity_valid_input(bioactivity_predictor):
    # Example SMILES for a known compound
    smiles = "CC(=O)Oc1ccccc(C(=O)O)cc1"
    result = bioactivity_predictor.predict_bioactivity(smiles)
    assert result is not None
    assert isinstance(result, dict)
    assert "activity_score" in result

def test_predict_bioactivity_invalid_input(bioactivity_predictor):
    invalid_smiles = "INVALID_SMILES"
    result = bioactivity_predictor.predict_bioactivity(invalid_smiles)
    assert result is None

def test_predict_bioactivity_edge_case(bioactivity_predictor):
    edge_case_smiles = "C1=CC=CC=C1"  # Benzene
    result = bioactivity_predictor.predict_bioactivity(edge_case_smiles)
    assert result is not None
    assert isinstance(result, dict)
    assert "activity_score" in result
    assert result["activity_score"] >= 0  # Assuming activity score is non-negative