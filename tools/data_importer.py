# -*- coding: utf-8 -*-
"""
Data Importer Tool
==================

This module is responsible for reading and filtering raw CSV files from
Rebrickable and creating a normalized SQLite database specifically
designed for this project.

Best Practices Followed:
- Uses `pathlib` for all file path manipulations to ensure cross-platform
  compatibility.
- Explicitly specifies `encoding='utf-8'` when reading all text files to
  prevent encoding issues.
"""

import pandas as pd
from pathlib import Path

class DataImporter:
    """
    A class responsible for processing Rebrickable data and building a local
    database.
    """
    def __init__(self, raw_data_path: str):
        """
        Initializes the DataImporter.

        Args:
            raw_data_path (str): The path to the directory containing the raw
                                 Rebrickable CSV files.
        """
        # Core Practice: Immediately convert the incoming path string into a
        # smart and safe Path object. This is the first step in our best
        # practice for handling special paths.
        self.raw_data_path = Path(raw_data_path)
        
        # Initialize DataFrames to store the loaded data
        self.sets_df = None
        self.parts_df = None
        self.colors_df = None
        self.inventories_df = None
        self.inventory_parts_df = None
        
        # Define the list of required CSV files
        self.required_files = [
            "sets.csv", "parts.csv", "colors.csv", 
            "inventories.csv", "inventory_parts.csv"
        ]

    def _load_csv_files(self):
        """
        Loads all required CSV files from the specified raw_data_path into
        pandas DataFrames.
        
        Raises:
            FileNotFoundError: If any of the required CSV files do not exist.
        """
        print(f"Loading CSV files from {self.raw_data_path}...")
        
        for filename in self.required_files:
            # Core Practice: Use the / operator of the Path object to safely
            # combine paths. This is safer than manual string concatenation
            # and automatically handles OS-specific separators.
            file_path = self.raw_data_path / filename
            
            if not file_path.exists():
                raise FileNotFoundError(f"Error: Required data file not found at '{file_path}'")

            # Store the loaded DataFrame into the corresponding instance attribute
            # based on the filename.
            df_name = filename.replace('.csv', '_df')

            # Core Practice: Explicitly specify encoding='utf-8' when reading CSVs.
            df = pd.read_csv(file_path, encoding='utf-8')
            setattr(self, df_name, df)
            
        print("All CSV files loaded successfully!")

    # --- Subsequent methods will be implemented here ---
    # def _filter_data(self):
    #     pass
    #
    # def _create_database(self):
    #     pass
    #
    # def run(self):
    #     self._load_csv_files()
    #     self._filter_data()
    #     self._create_database()

