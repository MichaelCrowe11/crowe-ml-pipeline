import logging

def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("universal_fungal_intelligence.log"),
            logging.StreamHandler()
        ]
    )

setup_logging()