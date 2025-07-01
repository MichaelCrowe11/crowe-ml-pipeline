import logging
import sys
import argparse
from core.fungal_intelligence import UniversalFungalIntelligence
from database import init_db
from utils.logging_config import setup_logging
from utils.bigquery_exporter import BigQueryExporter

def main(export_to_bigquery: bool = False):
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting the Universal Fungal Intelligence System...")
        
        # Initialize database
        logger.info("Initializing database...")
        init_db()
        
        # Create the fungal intelligence system
        fungal_intelligence_system = UniversalFungalIntelligence()
        
        # Initiate the analysis process
        logger.info("Beginning global fungal kingdom analysis...")
        analysis_results = fungal_intelligence_system.analyze_global_fungal_kingdom()
        
        # Log summary results
        logger.info("Analysis completed.")
        logger.info(f"Total species analyzed: {analysis_results.get('total_species_analyzed', 0)}")
        logger.info(f"Total compounds analyzed: {analysis_results.get('total_compounds_analyzed', 0)}")
        logger.info(f"Breakthrough discoveries: {len(analysis_results.get('breakthrough_discoveries', []))}")
        
        # Export to BigQuery if requested
        if export_to_bigquery:
            logger.info("Exporting results to BigQuery...")
            try:
                exporter = BigQueryExporter()
                exporter.create_tables_if_not_exist()
                job_id = exporter.export_analysis_results(analysis_results)
                logger.info(f"Successfully exported to BigQuery. Job ID: {job_id}")
                
                # Export metrics
                metrics = {
                    'dataset': 'fungal_analysis',
                    'n_train': 0,  # To be updated when training implemented
                    'n_test': 0,
                    'rmse': 0.0,
                    'r2': 0.0,
                    'model_type': 'fungal_intelligence',
                    'compound_analyzed': 'multiple',
                    'bioactivity_score': 0.0
                }
                exporter.export_metrics(metrics)
                
            except Exception as e:
                logger.error(f"Failed to export to BigQuery: {str(e)}")
                logger.info("Analysis results saved locally but not exported to cloud.")
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Universal Fungal Intelligence System')
    parser.add_argument(
        '--export-to-bigquery',
        action='store_true',
        help='Export analysis results to Google BigQuery'
    )
    parser.add_argument(
        '--analyze-compound',
        type=str,
        help='Analyze a specific compound (SMILES format)'
    )
    
    args = parser.parse_args()
    
    if args.analyze_compound:
        # Quick compound analysis mode
        from core.molecular_analyzer import MolecularAnalyzer
        analyzer = MolecularAnalyzer()
        result = analyzer.analyze_structure(args.analyze_compound)
        print(f"\nCompound Analysis Results:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        # Full system analysis
        main(export_to_bigquery=args.export_to_bigquery)