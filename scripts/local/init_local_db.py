#!/usr/bin/env python3
"""
Database Initialization Script
Initializes and populates the LEGO parts database from Rebrickable CSVs.
"""

import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.database import DatabaseManager, DataImporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Initializing database...")

    # Create database (tables)
    db = DatabaseManager()
    logger.info(f"Database created at: {db.db_path}")

    # Import data
    importer = DataImporter(db.db_path)
    importer.import_all()

    logger.info("Database initialization complete.")


if __name__ == "__main__":
    main()
