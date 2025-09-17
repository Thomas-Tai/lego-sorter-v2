# -*- coding: utf-8 -*-
"""
Execution script for the DataImporter tool.

This script initializes and runs the data import pipeline, which loads
raw data from Rebrickable, filters it for the target sets, and creates
a normalized SQLite database.
"""

from tools.data_importer import DataImporter

# --- Configuration ---
# This is the official, single source of truth for file paths in our project.
RAW_DATA_PATH = "data/raw/rebrickable_20250917"
DB_PATH = "data/processed/lego_parts.sqlite"
TARGET_SETS = ['45345-1']  # LEGO Spike Essential

def main():
    """
    Main function to execute the data import and processing pipeline.
    """
    print("=========================================")
    print(" Lego Sorter V2 - Data Import Pipeline ")
    print("=========================================")
    
    try:
        # 1. Initialize the Importer
        importer = DataImporter(raw_data_path=RAW_DATA_PATH, db_path=DB_PATH)
        
        # 2. Set the target
        importer.target_set_nums = TARGET_SETS
        
        # 3. Run the full processing pipeline
        importer.run()
        
        print("\n🎉 Database creation successful!")
        print(f"File saved to: {DB_PATH}")
        print("=========================================")

    except Exception as e:
        print(f"\n❌ An error occurred during processing: {e}")
        print("=========================================")

if __name__ == "__main__":
    main()
