import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sorter_app.data.database_manager import DatabaseManager
from sorter_app.data.data_importer import DataImporter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    db_path = os.path.join(os.path.dirname(__file__), '..', 'sorter_app', 'data', 'lego_parts.sqlite')
    raw_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'rebrickable_20250917')

    # Ensure data directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    logger.info(f"Initializing database at {db_path}")
    logger.info(f"Reading raw data from {raw_data_dir}")

    # 1. Create Tables
    db_manager = DatabaseManager(db_path)
    logger.info("Tables created.")

    # 2. Import Data
    importer = DataImporter(db_path, raw_data_dir)
    importer.import_all()
    
    logger.info("Database initialization complete.")

if __name__ == "__main__":
    main()
