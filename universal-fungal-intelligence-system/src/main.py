import logging
from core.fungal_intelligence import UniversalFungalIntelligence

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting the Universal Fungal Intelligence System...")
    
    fungal_intelligence_system = UniversalFungalIntelligence()
    
    # Initiate the analysis process
    analysis_results = fungal_intelligence_system.analyze_global_fungal_kingdom()
    
    logger.info("Analysis completed.")
    logger.info(f"Results: {analysis_results}")

if __name__ == "__main__":
    main()