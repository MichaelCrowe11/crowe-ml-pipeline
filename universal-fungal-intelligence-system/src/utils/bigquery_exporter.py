"""Export fungal analysis results to Google BigQuery."""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from google.cloud import bigquery
from google.cloud import storage
import os
import sys

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.gcp_config import (
    get_bigquery_config, get_gcs_config, get_full_table_id, get_gcs_uri
)

logger = logging.getLogger(__name__)

class BigQueryExporter:
    """Export analysis results to BigQuery."""
    
    def __init__(self):
        """Initialize BigQuery and GCS clients."""
        config = get_bigquery_config()
        self.project_id = config['project_id']
        self.dataset_id = config['dataset_id']
        
        # Initialize clients
        self.bq_client = bigquery.Client(project=self.project_id)
        self.storage_client = storage.Client(project=self.project_id)
        
        # Get bucket name
        gcs_config = get_gcs_config()
        self.bucket_name = gcs_config['bucket_name']
        
    def export_analysis_results(self, analysis_results: Dict[str, Any]) -> str:
        """
        Export fungal analysis results to BigQuery.
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            The BigQuery job ID
        """
        try:
            # Prepare data for BigQuery
            row_data = {
                'analysis_id': analysis_results.get('analysis_id'),
                'timestamp': analysis_results.get('timestamp'),
                'species_analyzed': analysis_results.get('total_species_analyzed', 0),
                'compounds_analyzed': analysis_results.get('total_compounds_analyzed', 0),
                'breakthrough_discoveries': len(analysis_results.get('breakthrough_discoveries', [])),
                'therapeutic_candidates': json.dumps(analysis_results.get('therapeutic_candidates', {})),
                'synthesis_pathways': json.dumps(analysis_results.get('synthesis_pathways', {})),
                'impact_assessment': json.dumps(analysis_results.get('impact_assessment', {}))
            }
            
            # Upload to GCS first
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            gcs_path = f"fungal_analysis/analysis_{timestamp}.json"
            self._upload_to_gcs(row_data, gcs_path)
            
            # Load into BigQuery
            table_id = get_full_table_id('fungal_analysis_results')
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                autodetect=False,
                write_disposition='WRITE_APPEND'
            )
            
            uri = get_gcs_uri(gcs_path)
            load_job = self.bq_client.load_table_from_uri(
                uri, table_id, job_config=job_config
            )
            
            load_job.result()  # Wait for job to complete
            logger.info(f"Successfully exported analysis results to BigQuery: {table_id}")
            
            return load_job.job_id
            
        except Exception as e:
            logger.error(f"Failed to export analysis results: {str(e)}")
            raise
    
    def export_compounds(self, compounds: List[Dict[str, Any]]) -> str:
        """
        Export discovered compounds to BigQuery.
        
        Args:
            compounds: List of compound dictionaries
            
        Returns:
            The BigQuery job ID
        """
        try:
            # Prepare compound data
            timestamp = datetime.now().isoformat()
            rows = []
            
            for compound in compounds:
                row = {
                    'compound_id': compound.get('id'),
                    'name': compound.get('name'),
                    'smiles': compound.get('smiles'),
                    'molecular_weight': compound.get('molecular_weight'),
                    'logp': compound.get('logP'),
                    'drug_likeness': compound.get('drug_likeness'),
                    'bioactivity_score': compound.get('bioactivity_score'),
                    'therapeutic_target': compound.get('therapeutic_target'),
                    'discovery_timestamp': timestamp,
                    'source_species': compound.get('source_species'),
                    'synthesis_feasibility': compound.get('synthesis_feasibility')
                }
                rows.append(row)
            
            # Upload to BigQuery
            table_id = get_full_table_id('discovered_compounds')
            errors = self.bq_client.insert_rows_json(table_id, rows)
            
            if errors:
                logger.error(f"Failed to insert compounds: {errors}")
                raise Exception(f"BigQuery insert failed: {errors}")
            
            logger.info(f"Successfully exported {len(compounds)} compounds to BigQuery")
            return f"Inserted {len(compounds)} compounds"
            
        except Exception as e:
            logger.error(f"Failed to export compounds: {str(e)}")
            raise
    
    def export_metrics(self, metrics: Dict[str, Any]) -> str:
        """
        Export ML model metrics to BigQuery.
        
        Args:
            metrics: Dictionary containing model metrics
            
        Returns:
            The BigQuery job ID
        """
        try:
            # Add timestamp if not present
            if 'timestamp' not in metrics:
                metrics['timestamp'] = datetime.now().isoformat()
            
            # Upload to GCS
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            gcs_path = f"ml_metrics/metrics_{timestamp}.jsonl"
            self._upload_to_gcs(metrics, gcs_path, as_jsonl=True)
            
            # Load into BigQuery
            table_id = get_full_table_id('metrics_log')
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                autodetect=False,
                write_disposition='WRITE_APPEND'
            )
            
            uri = get_gcs_uri(gcs_path)
            load_job = self.bq_client.load_table_from_uri(
                uri, table_id, job_config=job_config
            )
            
            load_job.result()
            logger.info(f"Successfully exported metrics to BigQuery: {table_id}")
            
            return load_job.job_id
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {str(e)}")
            raise
    
    def _upload_to_gcs(self, data: Dict[str, Any], path: str, as_jsonl: bool = True) -> None:
        """Upload data to Google Cloud Storage."""
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(path)
        
        if as_jsonl:
            content = json.dumps(data) + '\n'
        else:
            content = json.dumps(data, indent=2)
        
        blob.upload_from_string(content, content_type='application/json')
        logger.info(f"Uploaded data to GCS: gs://{self.bucket_name}/{path}")
    
    def create_tables_if_not_exist(self) -> None:
        """Create BigQuery tables if they don't exist."""
        dataset_id = f"{self.project_id}.{self.dataset_id}"
        
        # Create dataset if not exists
        try:
            self.bq_client.get_dataset(dataset_id)
        except:
            dataset = bigquery.Dataset(dataset_id)
            dataset = self.bq_client.create_dataset(dataset, exists_ok=True)
            logger.info(f"Created dataset: {dataset_id}")
        
        # Define table schemas
        schemas = {
            'discovered_compounds': [
                bigquery.SchemaField("compound_id", "STRING"),
                bigquery.SchemaField("name", "STRING"),
                bigquery.SchemaField("smiles", "STRING"),
                bigquery.SchemaField("molecular_weight", "FLOAT"),
                bigquery.SchemaField("logp", "FLOAT"),
                bigquery.SchemaField("drug_likeness", "STRING"),
                bigquery.SchemaField("bioactivity_score", "FLOAT"),
                bigquery.SchemaField("therapeutic_target", "STRING"),
                bigquery.SchemaField("discovery_timestamp", "TIMESTAMP"),
                bigquery.SchemaField("source_species", "STRING"),
                bigquery.SchemaField("synthesis_feasibility", "FLOAT"),
            ]
        }
        
        # Create tables
        for table_name, schema in schemas.items():
            table_id = get_full_table_id(table_name)
            
            try:
                self.bq_client.get_table(table_id)
                logger.info(f"Table already exists: {table_id}")
            except:
                table = bigquery.Table(table_id, schema=schema)
                table = self.bq_client.create_table(table)
                logger.info(f"Created table: {table_id}") 