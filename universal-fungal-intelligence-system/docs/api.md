# Universal Fungal Intelligence System API Documentation

## Overview

The Universal Fungal Intelligence System provides a comprehensive API for analyzing fungal species and their chemical compounds. This documentation outlines the available endpoints, their functionalities, and usage examples.

## Base URL

The base URL for the API is:

```
http://localhost:8000/api
```

## Endpoints

### 1. Species Analysis

#### GET /species

Retrieve a list of all documented fungal species.

**Response:**
- 200 OK
- Returns a JSON array of species objects.

#### Example Request:
```
GET /species
```

#### Example Response:
```json
[
    {
        "id": 1,
        "scientific_name": "Aspergillus niger",
        "genus": "Aspergillus",
        "phylum": "Ascomycota"
    },
    {
        "id": 2,
        "scientific_name": "Penicillium chrysogenum",
        "genus": "Penicillium",
        "phylum": "Ascomycota"
    }
]
```

### 2. Compound Analysis

#### GET /compounds

Retrieve a list of all known compounds.

**Response:**
- 200 OK
- Returns a JSON array of compound objects.

#### Example Request:
```
GET /compounds
```

#### Example Response:
```json
[
    {
        "id": 1,
        "name": "Penicillin G",
        "type": "antibiotic"
    },
    {
        "id": 2,
        "name": "Fumigaclavine C",
        "type": "alkaloid"
    }
]
```

### 3. Chemical Analysis

#### POST /analysis/chemical

Analyze the chemical composition of a specific fungal species.

**Request Body:**
- JSON object containing the species ID.

#### Example Request:
```
POST /analysis/chemical
Content-Type: application/json

{
    "species_id": 1
}
```

**Response:**
- 200 OK
- Returns the chemical profile of the species.

#### Example Response:
```json
{
    "species_id": 1,
    "chemical_profile": {
        "primary_metabolites": ["citric_acid", "gluconic_acid"],
        "secondary_metabolites": ["fumigaclavine_C"]
    }
}
```

### 4. Therapeutic Assessment

#### POST /analysis/therapeutic

Assess the therapeutic potential of a specific compound.

**Request Body:**
- JSON object containing the compound ID.

#### Example Request:
```
POST /analysis/therapeutic
Content-Type: application/json

{
    "compound_id": 1
}
```

**Response:**
- 200 OK
- Returns the therapeutic assessment results.

#### Example Response:
```json
{
    "compound_id": 1,
    "therapeutic_potential": "high",
    "mechanisms": ["antibacterial", "anti-inflammatory"]
}
```

## Error Handling

The API returns standard HTTP status codes to indicate the success or failure of a request. Common error responses include:

- 400 Bad Request: The request was invalid.
- 404 Not Found: The requested resource was not found.
- 500 Internal Server Error: An error occurred on the server.

## Conclusion

This API provides a powerful interface for accessing and analyzing fungal species and their compounds. For further information or assistance, please refer to the other documentation files or contact the development team.