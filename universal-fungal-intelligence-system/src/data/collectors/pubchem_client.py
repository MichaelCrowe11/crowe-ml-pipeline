import requests
import time
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import quote

logger = logging.getLogger(__name__)

class PubChemClient:
    """
    Client for interacting with the PubChem REST API.
    Documentation: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
    """
    
    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    COMPOUND_URL = f"{BASE_URL}/compound"
    SUBSTANCE_URL = f"{BASE_URL}/substance"
    
    def __init__(self, rate_limit_delay: float = 0.2):
        """
        Initialize PubChem client.
        
        Args:
            rate_limit_delay: Delay between API requests in seconds (5 requests/second max)
        """
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'UniversalFungalIntelligenceSystem/1.0'
        })
    
    def get_compound_by_cid(self, cid: int) -> Dict[str, Any]:
        """
        Get compound information by PubChem CID.
        
        Args:
            cid: PubChem Compound ID
            
        Returns:
            Compound information dictionary
        """
        try:
            # Get basic properties
            properties_url = f"{self.COMPOUND_URL}/cid/{cid}/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IUPACName,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount/JSON"
            response = self._make_request(properties_url)
            
            if response and 'PropertyTable' in response:
                properties = response['PropertyTable']['Properties'][0]
                
                # Get additional bioactivity data
                bioactivity = self._get_bioactivity_data(cid)
                
                return {
                    'cid': cid,
                    'name': properties.get('IUPACName', ''),
                    'molecular_formula': properties.get('MolecularFormula', ''),
                    'molecular_weight': properties.get('MolecularWeight', 0),
                    'smiles': properties.get('CanonicalSMILES', ''),
                    'logp': properties.get('XLogP', 0),
                    'tpsa': properties.get('TPSA', 0),
                    'h_bond_donors': properties.get('HBondDonorCount', 0),
                    'h_bond_acceptors': properties.get('HBondAcceptorCount', 0),
                    'bioactivity': bioactivity
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching compound {cid}: {str(e)}")
            return {}
    
    def get_compound_by_name(self, name: str) -> Dict[str, Any]:
        """
        Get compound information by name.
        
        Args:
            name: Compound name
            
        Returns:
            Compound information dictionary
        """
        try:
            # Search for compound by name
            search_url = f"{self.COMPOUND_URL}/name/{quote(name)}/cids/JSON"
            response = self._make_request(search_url)
            
            if response and 'IdentifierList' in response:
                cids = response['IdentifierList']['CID']
                if cids:
                    # Get data for the first matching compound
                    return self.get_compound_by_cid(cids[0])
            
            return {}
            
        except Exception as e:
            logger.error(f"Error searching for compound '{name}': {str(e)}")
            return {}
    
    def get_compounds_by_formula(self, formula: str) -> List[Dict[str, Any]]:
        """
        Get compounds by molecular formula.
        
        Args:
            formula: Molecular formula (e.g., "C9H8O4" for aspirin)
            
        Returns:
            List of compound information dictionaries
        """
        try:
            # Search by formula
            search_url = f"{self.COMPOUND_URL}/formula/{formula}/cids/JSON"
            response = self._make_request(search_url)
            
            compounds = []
            if response and 'IdentifierList' in response:
                cids = response['IdentifierList']['CID'][:10]  # Limit to 10 results
                
                for cid in cids:
                    compound_data = self.get_compound_by_cid(cid)
                    if compound_data:
                        compounds.append(compound_data)
                    time.sleep(self.rate_limit_delay)
            
            return compounds
            
        except Exception as e:
            logger.error(f"Error searching by formula '{formula}': {str(e)}")
            return []
    
    def get_compound_by_smiles(self, smiles: str) -> Dict[str, Any]:
        """
        Get compound information by SMILES string.
        
        Args:
            smiles: SMILES representation
            
        Returns:
            Compound information dictionary
        """
        try:
            # Search by SMILES
            search_url = f"{self.COMPOUND_URL}/smiles/{quote(smiles)}/cids/JSON"
            response = self._make_request(search_url)
            
            if response and 'IdentifierList' in response:
                cids = response['IdentifierList']['CID']
                if cids:
                    return self.get_compound_by_cid(cids[0])
            
            return {}
            
        except Exception as e:
            logger.error(f"Error searching by SMILES: {str(e)}")
            return {}
    
    def search_fungal_compounds(self, query: str = "fungal metabolite", max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Search for fungal-related compounds.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of compound information dictionaries
        """
        try:
            # Use PubChem's text search
            search_url = f"{self.BASE_URL}/compound/name/{quote(query)}/cids/JSON?name_type=word"
            response = self._make_request(search_url)
            
            compounds = []
            if response and 'IdentifierList' in response:
                cids = response['IdentifierList']['CID'][:max_results]
                
                logger.info(f"Found {len(cids)} compounds for query '{query}'")
                
                for i, cid in enumerate(cids):
                    if i % 10 == 0:
                        logger.info(f"Processing compound {i+1}/{len(cids)}")
                    
                    compound_data = self.get_compound_by_cid(cid)
                    if compound_data:
                        compounds.append(compound_data)
                    
                    time.sleep(self.rate_limit_delay)
            
            return compounds
            
        except Exception as e:
            logger.error(f"Error searching fungal compounds: {str(e)}")
            return []
    
    def _get_bioactivity_data(self, cid: int) -> Dict[str, Any]:
        """Get bioactivity data for a compound."""
        try:
            # Get bioassay data
            assay_url = f"{self.BASE_URL}/compound/cid/{cid}/assaysummary/JSON"
            response = self._make_request(assay_url)
            
            bioactivity = {
                'active_assays': 0,
                'inactive_assays': 0,
                'inconclusive_assays': 0,
                'total_assays': 0
            }
            
            if response and 'Table' in response:
                rows = response['Table'].get('Row', [])
                for row in rows:
                    cells = row.get('Cell', [])
                    if len(cells) >= 4:
                        bioactivity['active_assays'] = int(cells[0])
                        bioactivity['inactive_assays'] = int(cells[1])
                        bioactivity['inconclusive_assays'] = int(cells[2])
                        bioactivity['total_assays'] = int(cells[3])
            
            return bioactivity
            
        except Exception as e:
            logger.debug(f"Could not fetch bioactivity data for CID {cid}: {str(e)}")
            return {}
    
    def _make_request(self, url: str) -> Optional[Dict]:
        """Make HTTP request with error handling and rate limiting."""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for URL {url}: {str(e)}")
            return None
        except ValueError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            return None
    
    def close(self):
        """Close the session."""
        self.session.close()