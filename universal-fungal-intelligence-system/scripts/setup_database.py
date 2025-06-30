import os
import sys
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.models import Base

logger = logging.getLogger(__name__)

def setup_database(db_url: str):
    """Set up the database by creating the necessary tables."""
    try:
        # Create a database engine
        engine = create_engine(db_url)
        
        # Create all tables in the database
        Base.metadata.create_all(engine)
        
        logger.info("Database setup completed successfully.")
    except Exception as e:
        logger.error(f"Error setting up the database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Example database URL, replace with actual configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///universal_fungal_intelligence.db")
    
    setup_database(DATABASE_URL)