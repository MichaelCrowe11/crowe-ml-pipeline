def calculate_molecular_weight(compound: str) -> float:
    """Calculate the molecular weight of a given compound."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(compound)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")
    return Chem.rdMolDescriptors.CalcExactMolWt(mol)

def predict_logp(compound: str) -> float:
    """Predict the logP (partition coefficient) of a given compound."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(compound)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")
    return Chem.Crippen.MolLogP(mol)

def get_tpsa(compound: str) -> float:
    """Calculate the Topological Polar Surface Area (TPSA) of a given compound."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(compound)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")
    return Chem.rdMolDescriptors.CalcTPSA(mol)

def generate_fingerprint(compound: str) -> str:
    """Generate a molecular fingerprint for a given compound."""
    from rdkit import Chem
    from rdkit.Chem import RDKFingerprint
    mol = Chem.MolFromSmiles(compound)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")
    fingerprint = RDKFingerprint(mol)
    return fingerprint.ToBitString()