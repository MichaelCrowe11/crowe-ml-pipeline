import requests

class NCBIClient:
    """
    A client for interacting with the NCBI API to collect fungal data.
    """

    BASE_URL = "https://api.ncbi.nlm.nih.gov/"

    def __init__(self):
        """Initialize the NCBI client."""
        self.session = requests.Session()

    def search_fungi(self, query: str, max_results: int = 20) -> dict:
        """
        Search for fungal species in the NCBI database.

        Args:
            query (str): The search term for fungal species.
            max_results (int): The maximum number of results to return.

        Returns:
            dict: The search results from the NCBI API.
        """
        endpoint = f"{self.BASE_URL}taxonomy/v1/search"
        params = {
            "query": query,
            "max_results": max_results
        }
        response = self.session.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()

    def fetch_species_details(self, species_id: str) -> dict:
        """
        Fetch detailed information about a specific fungal species.

        Args:
            species_id (str): The NCBI species ID.

        Returns:
            dict: Detailed information about the species.
        """
        endpoint = f"{self.BASE_URL}taxonomy/v1/{species_id}"
        response = self.session.get(endpoint)
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the session."""
        self.session.close()