from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen
import logging

logger = logging.getLogger(__name__)

class MolecularAnalyzer:
    """
    Class for analyzing molecular structures of fungal compounds using RDKit.
    """

    def __init__(self):
        """Initialize the MolecularAnalyzer."""
        self.lipinski_rules = {
            'molecular_weight': (150, 500),
            'logP': (-0.4, 5.6),
            'h_donors': (0, 5),
            'h_acceptors': (0, 10)
        }

    def analyze_structure(self, smiles: str) -> dict:
        """
        Analyze the molecular structure given a SMILES representation.

        Args:
            smiles (str): The SMILES representation of the compound.

        Returns:
            dict: A dictionary containing molecular properties and descriptors.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error(f"Invalid SMILES: {smiles}")
                return {'error': 'Invalid SMILES string'}
            
            molecular_properties = {
                'smiles': smiles,
                'molecular_weight': self.calculate_molecular_weight(mol),
                'logP': self.calculate_logP(mol),
                'num_h_donors': self.calculate_num_h_donors(mol),
                'num_h_acceptors': self.calculate_num_h_acceptors(mol),
                'tpsa': self.calculate_tpsa(mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
                'lipinski_violations': self.check_lipinski_violations(mol),
                'drug_likeness': self.assess_drug_likeness(mol)
            }
            return molecular_properties
            
        except Exception as e:
            logger.error(f"Error analyzing structure: {e}")
            return {'error': str(e)}

    def calculate_molecular_weight(self, mol: Chem.Mol) -> float:
        """Calculate the molecular weight of the compound."""
        return Descriptors.ExactMolWt(mol)

    def calculate_logP(self, mol: Chem.Mol) -> float:
        """Calculate the logP (partition coefficient) of the compound."""
        return Crippen.MolLogP(mol)

    def calculate_num_h_donors(self, mol: Chem.Mol) -> int:
        """Calculate the number of hydrogen donors in the compound."""
        return Lipinski.NumHDonors(mol)

    def calculate_num_h_acceptors(self, mol: Chem.Mol) -> int:
        """Calculate the number of hydrogen acceptors in the compound."""
        return Lipinski.NumHAcceptors(mol)

    def calculate_tpsa(self, mol: Chem.Mol) -> float:
        """Calculate the topological polar surface area (TPSA) of the compound."""
        return Descriptors.TPSA(mol)
    
    def check_lipinski_violations(self, mol: Chem.Mol) -> int:
        """Check Lipinski's Rule of Five violations."""
        violations = 0
        
        mw = self.calculate_molecular_weight(mol)
        if not (self.lipinski_rules['molecular_weight'][0] <= mw <= self.lipinski_rules['molecular_weight'][1]):
            violations += 1
            
        logp = self.calculate_logP(mol)
        if not (self.lipinski_rules['logP'][0] <= logp <= self.lipinski_rules['logP'][1]):
            violations += 1
            
        h_donors = self.calculate_num_h_donors(mol)
        if not (self.lipinski_rules['h_donors'][0] <= h_donors <= self.lipinski_rules['h_donors'][1]):
            violations += 1
            
        h_acceptors = self.calculate_num_h_acceptors(mol)
        if not (self.lipinski_rules['h_acceptors'][0] <= h_acceptors <= self.lipinski_rules['h_acceptors'][1]):
            violations += 1
            
        return violations
    
    def assess_drug_likeness(self, mol: Chem.Mol) -> str:
        """Assess the drug-likeness of the compound."""
        violations = self.check_lipinski_violations(mol)
        
        if violations == 0:
            return "Excellent"
        elif violations == 1:
            return "Good"
        elif violations == 2:
            return "Moderate"
        else:
            return "Poor"