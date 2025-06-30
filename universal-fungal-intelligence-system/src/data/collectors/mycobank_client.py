from typing import Dict, Any
import aiohttp

class MycoBankClient:
    """
    Client for interacting with the MycoBank API to collect fungal data.
    """

    BASE_URL = "https://www.mycobank.org/api"

    async def fetch_species_data(self, species_name: str) -> Dict[str, Any]:
        """
        Fetch data for a specific fungal species from MycoBank.

        Args:
            species_name (str): The name of the species to fetch data for.

        Returns:
            Dict[str, Any]: The data for the specified species.
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.BASE_URL}/species/{species_name}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Error fetching data for {species_name}: {response.status}")

    async def fetch_all_species(self) -> Dict[str, Any]:
        """
        Fetch data for all fungal species from MycoBank.

        Returns:
            Dict[str, Any]: The data for all species.
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.BASE_URL}/species") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Error fetching all species data: {response.status}")

    async def fetch_compound_data(self, compound_name: str) -> Dict[str, Any]:
        """
        Fetch data for a specific compound from MycoBank.

        Args:
            compound_name (str): The name of the compound to fetch data for.

        Returns:
            Dict[str, Any]: The data for the specified compound.
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.BASE_URL}/compounds/{compound_name}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Error fetching data for {compound_name}: {response.status}")