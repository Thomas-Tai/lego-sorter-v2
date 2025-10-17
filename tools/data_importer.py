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
from typing import List, Optional
import sqlite3


class DataImporter:
    """
    A class responsible for processing Rebrickable data and building a local
    database.
    """

    def __init__(self, raw_data_path: str, db_path: str):
        """
        Initializes the DataImporter.

        Args:
            raw_data_path (str): The path to the directory containing the raw
                                 Rebrickable CSV files.
            db_path (str): The path where the output SQLite database will be saved.
        """
        self.raw_data_path = Path(raw_data_path)
        self.db_path = Path(db_path)
        self.target_set_nums: List[str] = []

        self.sets_df: Optional[pd.DataFrame] = None
        self.parts_df: Optional[pd.DataFrame] = None
        self.colors_df: Optional[pd.DataFrame] = None
        self.inventories_df: Optional[pd.DataFrame] = None
        self.inventory_parts_df: Optional[pd.DataFrame] = None

        self.required_files = [
            "sets.csv",
            "parts.csv",
            "colors.csv",
            "inventories.csv",
            "inventory_parts.csv",
        ]

    def _load_csv_files(self):
        """
        Loads all required CSV files from the specified raw_data_path into
        pandas DataFrames.
        """
        print(f"Loading CSV files from {self.raw_data_path}...")
        for filename in self.required_files:
            file_path = self.raw_data_path / filename
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Error: Required data file not found at '{file_path}'"
                )
            df_name = filename.replace(".csv", "_df")
            df = pd.read_csv(file_path, encoding="utf-8")
            setattr(self, df_name, df)
        print("All CSV files loaded successfully!")

    def _filter_data(self):
        """
        Filters all loaded DataFrames to only contain data relevant to the
        target_set_nums. This version is hardened to handle real-world data.
        """
        if not self.target_set_nums:
            print("Warning: No target set numbers specified. Skipping filtering.")
            return

        print(f"Filtering data for target sets: {self.target_set_nums}...")

        # --- DEFENSIVE DATA TYPING ---
        # Ensure join keys are of the same type to prevent silent merge failures.
        self.sets_df["set_num"] = self.sets_df["set_num"].astype(str)
        self.inventories_df["set_num"] = self.inventories_df["set_num"].astype(str)
        self.inventory_parts_df["inventory_id"] = self.inventory_parts_df[
            "inventory_id"
        ].astype(int)
        self.inventories_df["id"] = self.inventories_df["id"].astype(int)

        # --- STEP 1: Find the inventory IDs for our target sets ---
        # For each set, we only want the latest version of the inventory.
        # This prevents us from including parts from older, outdated inventories.
        latest_inventories = self.inventories_df.loc[
            self.inventories_df.groupby("set_num")["version"].idxmax()
        ]

        # Now, merge this cleaned inventory list with our target sets.
        target_inventories = pd.merge(
            latest_inventories,
            self.sets_df[self.sets_df["set_num"].isin(self.target_set_nums)],
            on="set_num",
            how="inner",
        )

        if target_inventories.empty:
            raise ValueError(
                f"Could not find any inventories for target sets: {self.target_set_nums}"
            )

        target_inventory_ids = target_inventories["id"].unique().tolist()

        # --- STEP 2: Filter the main inventory_parts DataFrame ---
        # This is the most critical filtering step.
        self.inventory_parts_df = self.inventory_parts_df[
            self.inventory_parts_df["inventory_id"].isin(target_inventory_ids)
        ].copy()  # Use .copy() to avoid SettingWithCopyWarning in pandas

        # --- STEP 3: Use the filtered inventory to filter other DataFrames ---
        relevant_part_nums = self.inventory_parts_df["part_num"].unique()
        relevant_color_ids = self.inventory_parts_df["color_id"].unique()

        # Filter parts and colors DataFrames
        self.parts_df = self.parts_df[
            self.parts_df["part_num"].isin(relevant_part_nums)
        ].copy()
        self.colors_df = self.colors_df[
            self.colors_df["id"].isin(relevant_color_ids)
        ].copy()

        print("Data filtering complete!")

    def _create_database(self):
        """
        Writes the filtered DataFrames into a new, clean SQLite database file.
        If the file already exists, it will be overwritten.
        """
        print(f"Creating database at {self.db_path}...")

        # Ensure the parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a connection. This will create the file if it doesn't exist.
        conn = sqlite3.connect(self.db_path)

        try:
            # Use pandas' .to_sql() to write DataFrames to tables.
            # `if_exists='replace'` will drop the table first if it exists.
            # `index=False` prevents pandas from writing the DataFrame index as a column.
            self.inventory_parts_df.to_sql(
                "inventory_parts", conn, if_exists="replace", index=False
            )

            # --- SCHEMA EVOLUTION: Add the image_folder_name column ---
            # We add the new column to the DataFrame before writing it to SQL.
            # Pandas will automatically create this column in the SQL table.
            # A value of None in pandas becomes a NULL value in SQL.
            print("Adding 'image_folder_name' column to 'parts' table schema...")
            self.parts_df["image_folder_name"] = None
            # -----------------------------------------------------------

            self.parts_df.to_sql("parts", conn, if_exists="replace", index=False)
            self.colors_df.to_sql("colors", conn, if_exists="replace", index=False)

            # We also save the filtered sets table for context
            filtered_sets_df = self.sets_df[
                self.sets_df["set_num"].isin(self.target_set_nums)
            ].copy()
            filtered_sets_df.to_sql("sets", conn, if_exists="replace", index=False)

            print("Database and tables created successfully.")

        finally:
            # Ensure the connection is always closed.
            conn.close()

    def run(self):
        """
        Executes the full data import pipeline from loading CSVs to creating
        the final SQLite database.
        """
        self._load_csv_files()
        self._filter_data()
        self._create_database()
