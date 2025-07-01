import pytest
from src.core.molecular_analyzer import MolecularAnalyzer

class TestMolecularAnalyzer:
    """Test cases for the MolecularAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = MolecularAnalyzer()
        
        # Test compounds
        self.test_compounds = {
            'aspirin': 'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
            'caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'penicillin': 'CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O',  # Penicillin G
            'invalid': 'INVALID_SMILES_STRING'
        }
    
    def test_analyze_structure_valid(self):
        """Test analyzing valid molecular structures."""
        result = self.analyzer.analyze_structure(self.test_compounds['aspirin'])
        
        assert 'molecular_weight' in result
        assert 'logP' in result
        assert 'num_h_donors' in result
        assert 'num_h_acceptors' in result
        assert 'tpsa' in result
        assert 'drug_likeness' in result
        
        # Check aspirin properties
        assert 175 < result['molecular_weight'] < 185  # ~180.16
        assert result['num_h_donors'] == 1
        assert result['num_h_acceptors'] == 4
    
    def test_analyze_structure_invalid(self):
        """Test handling of invalid SMILES strings."""
        result = self.analyzer.analyze_structure(self.test_compounds['invalid'])
        assert 'error' in result
        assert result['error'] == 'Invalid SMILES string'
    
    def test_lipinski_violations(self):
        """Test Lipinski rule violation checking."""
        # Test with caffeine (should have no violations)
        result = self.analyzer.analyze_structure(self.test_compounds['caffeine'])
        assert result['lipinski_violations'] == 0
        assert result['drug_likeness'] == 'Excellent'
    
    def test_drug_likeness_assessment(self):
        """Test drug-likeness assessment for different compounds."""
        compounds_to_test = ['aspirin', 'caffeine', 'penicillin']
        
        for compound_name in compounds_to_test:
            smiles = self.test_compounds[compound_name]
            result = self.analyzer.analyze_structure(smiles)
            
            assert 'drug_likeness' in result
            assert result['drug_likeness'] in ['Excellent', 'Good', 'Moderate', 'Poor']

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