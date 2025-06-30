import pytest
from src.data.collectors.mycobank_client import MycoBankClient
from src.data.collectors.ncbi_client import NCBIClient
from src.data.collectors.pubchem_client import PubChemClient

@pytest.fixture
def mycobank_client():
    return MycoBankClient()

@pytest.fixture
def ncbi_client():
    return NCBIClient()

@pytest.fixture
def pubchem_client():
    return PubChemClient()

def test_collect_fungal_data(mycobank_client):
    species_data = mycobank_client.collect_species_data()
    assert isinstance(species_data, dict)
    assert 'species' in species_data

def test_collect_ncbi_data(ncbi_client):
    ncbi_data = ncbi_client.collect_data()
    assert isinstance(ncbi_data, dict)
    assert 'fungi' in ncbi_data

def test_collect_pubchem_data(pubchem_client):
    pubchem_data = pubchem_client.collect_data()
    assert isinstance(pubchem_data, dict)
    assert 'compounds' in pubchem_data

def test_data_collection_integration(mycobank_client, ncbi_client, pubchem_client):
    species_data = mycobank_client.collect_species_data()
    ncbi_data = ncbi_client.collect_data()
    pubchem_data = pubchem_client.collect_data()

    assert isinstance(species_data, dict)
    assert isinstance(ncbi_data, dict)
    assert isinstance(pubchem_data, dict)

    assert 'species' in species_data
    assert 'fungi' in ncbi_data
    assert 'compounds' in pubchem_data

    # Further integration checks can be added here based on the expected structure of the data.