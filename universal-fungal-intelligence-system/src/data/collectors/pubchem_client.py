import requests

class PubChemClient:
    """
    Client for interacting with the PubChem API to collect chemical data.
    """

    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    @staticmethod
    def get_compound_by_cid(cid: int) -> dict:
        """
        Retrieve compound information from PubChem using the compound ID (CID).

        Args:
            cid (int): The compound ID.

        Returns:
            dict: The compound information.
        """
        url = f"{PubChemClient.BASE_URL}/compound/CID/{cid}/JSON"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def get_compound_by_name(name: str) -> dict:
        """
        Retrieve compound information from PubChem using the compound name.

        Args:
            name (str): The name of the compound.

        Returns:
            dict: The compound information.
        """
        url = f"{PubChemClient.BASE_URL}/compound/name/{name}/JSON"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def get_compounds_by_formula(formula: str) -> dict:
        """
        Retrieve compounds from PubChem using the molecular formula.

        Args:
            formula (str): The molecular formula.

        Returns:
            dict: The compounds information.
        """
        url = f"{PubChemClient.BASE_URL}/compound/formula/{formula}/JSON"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()