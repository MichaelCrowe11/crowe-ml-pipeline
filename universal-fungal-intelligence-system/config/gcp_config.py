"""Google Cloud Platform configuration for the Universal Fungal Intelligence System."""

import os
from typing import Dict, Any

# GCP Project Configuration
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'crowechem-fungi')
GCP_BUCKET_NAME = os.getenv('GCP_BUCKET_NAME', 'crowechem-fungi-ml-metrics')
GCP_REGION = os.getenv('GCP_REGION', 'us-central1')

# BigQuery Configuration
BIGQUERY_DATASET = 'crowe_ml_pipeline'
BIGQUERY_METRICS_TABLE = 'metrics_log'
BIGQUERY_FUNGAL_ANALYSIS_TABLE = 'fungal_analysis_results'
BIGQUERY_COMPOUNDS_TABLE = 'discovered_compounds'
BIGQUERY_SPECIES_TABLE = 'analyzed_species'

# GCS paths
GCS_METRICS_PREFIX = 'ml_metrics/'
GCS_ANALYSIS_PREFIX = 'fungal_analysis/'
GCS_COMPOUNDS_PREFIX = 'compounds/'

def get_bigquery_config() -> Dict[str, Any]:
    """Get BigQuery configuration."""
    return {
        'project_id': GCP_PROJECT_ID,
        'dataset_id': BIGQUERY_DATASET,
        'location': GCP_REGION,
        'tables': {
            'metrics': BIGQUERY_METRICS_TABLE,
            'fungal_analysis': BIGQUERY_FUNGAL_ANALYSIS_TABLE,
            'compounds': BIGQUERY_COMPOUNDS_TABLE,
            'species': BIGQUERY_SPECIES_TABLE
        }
    }

def get_gcs_config() -> Dict[str, Any]:
    """Get Google Cloud Storage configuration."""
    return {
        'bucket_name': GCP_BUCKET_NAME,
        'project_id': GCP_PROJECT_ID,
        'paths': {
            'metrics': GCS_METRICS_PREFIX,
            'analysis': GCS_ANALYSIS_PREFIX,
            'compounds': GCS_COMPOUNDS_PREFIX
        }
    }

def get_full_table_id(table_name: str) -> str:
    """Get full BigQuery table ID."""
    return f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{table_name}"

def get_gcs_uri(path: str) -> str:
    """Get full GCS URI."""
    return f"gs://{GCP_BUCKET_NAME}/{path}" 