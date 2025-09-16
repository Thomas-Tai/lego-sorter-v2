# -*- coding: utf-8 -*-
"""
Unit tests for the DataImporter tool.
"""

import pytest
import pandas as pd
from pathlib import Path
import sqlite3
from tools.data_importer import DataImporter


# --- Fixture for creating a reusable importer instance ---
@pytest.fixture
def configured_importer(tmp_path):
    """
    A pytest fixture that sets up a DataImporter instance with mock CSV files.
    This avoids code duplication in tests.
    """
    raw_data_path = tmp_path / "raw"
    raw_data_path.mkdir()
    
    # THE FIX: Define a dummy database path for our tests
    db_path = tmp_path / "test_lego_parts.sqlite"

    # Create mock CSVs...
    (raw_data_path / "sets.csv").write_text("set_num,name\n45345-1,Spike Essential\n99999-1,Other Set", encoding='utf-8')
    (raw_data_path / "inventories.csv").write_text("id,version,set_num\n123,1,45345-1\n456,1,99999-1", encoding='utf-8')
    (raw_data_path / "inventory_parts.csv").write_text("inventory_id,part_num,color_id,quantity\n123,3001,4,10\n123,3002,0,5\n456,9999,1,1", encoding='utf-8')
    (raw_data_path / "parts.csv").write_text("part_num,name\n3001,Brick 2x4\n3002,Plate 1x1\n9999,Alien Head", encoding='utf-8')
    (raw_data_path / "colors.csv").write_text("id,name\n4,Red\n0,Black\n1,Blue", encoding='utf-8')

    importer = DataImporter(raw_data_path=str(raw_data_path), db_path=str(db_path))
    importer.target_set_nums = ['45345-1']
    importer._load_csv_files()
    importer._filter_data()
    return importer


# --- Existing Tests (now using the fixture) ---


def test_load_csv_files_success(configured_importer):
    """
    Tests if the _load_csv_files method successfully loads all required CSV
    files into non-empty pandas DataFrames.
    """
    importer = configured_importer
    assert isinstance(importer.sets_df, pd.DataFrame) and not importer.sets_df.empty
    assert not importer.sets_df.empty
    assert isinstance(importer.parts_df, pd.DataFrame) and not importer.parts_df.empty
    assert not importer.parts_df.empty
    assert isinstance(importer.colors_df, pd.DataFrame) and not importer.colors_df.empty
    assert not importer.colors_df.empty
    assert isinstance(importer.inventories_df, pd.DataFrame) and not importer.inventories_df.empty
    assert not importer.inventories_df.empty
    assert isinstance(importer.inventory_parts_df, pd.DataFrame) and not importer.inventory_parts_df.empty
    assert not importer.inventory_parts_df.empty


def test_load_csv_files_file_not_found(tmp_path):
    """
    Tests that a FileNotFoundError is correctly raised when a required CSV
    file is missing.
    """
    raw_data_path = tmp_path / "raw"
    raw_data_path.mkdir()
    # THE FIX: Define a dummy database path
    db_path = tmp_path / "dummy.sqlite"
    
    # THE FIX: Pass the new 'db_path' argument
    importer = DataImporter(raw_data_path=str(raw_data_path), db_path=str(db_path))

    with pytest.raises(FileNotFoundError):
        importer._load_csv_files()


def test_filter_data_isolates_target_set_parts(configured_importer):
    """
    Tests if the _filter_data method correctly filters the DataFrames to
    contain only the parts relevant to the target set ('45345-1').
    """
    # --- 1. Arrange ---
    # The 'configured_importer' fixture has already done the arrangement for us.
    # It loaded data for two sets: '45345-1' and '99999-1'.
    importer = configured_importer
    assert len(importer.inventory_parts_df) == 2

    # --- 2. Act ---
    # Call the new method we are about to develop.
    importer._filter_data()

    # --- 3. Assert ---
    # Verify that the main inventory DataFrame now only contains parts from our target inventory (id 123).
    # All parts from the 'Other Set' (inventory_id 456) should be gone.
    assert importer.inventory_parts_df is not None
    assert (
        len(importer.inventory_parts_df) == 2
    ), "Should only contain the 2 parts from Spike Essential"
    assert importer.inventory_parts_df["inventory_id"].unique().tolist() == [123]

    # Optional: Verify that other related DataFrames are also filtered
    # For example, parts_df should now only contain '3001' and '3002'
    assert len(importer.parts_df) == 2
    assert "9999" not in importer.parts_df["part_num"].values

# --- NEW TEST CASE for the database creation feature ---

def test_create_database_writes_filtered_data_to_sqlite(configured_importer):
    """
    Tests if the _create_database method correctly writes the filtered
    DataFrames to a new SQLite database file.
    """
    # --- 1. Arrange ---
    importer = configured_importer
    # For this test, we override the db_path to be predictable
    db_path_for_test = configured_importer.raw_data_path.parent / "test_db.sqlite"
    importer.db_path = db_path_for_test

    # --- 2. Act ---
    # Call the new method we are about to develop.
    importer._create_database()

    # --- 3. Assert ---
    # Assertion 1: Check if the database file was actually created.
    assert db_path_for_test.exists(), "The SQLite database file should be created."

    # Assertion 2: Connect to the new DB and verify its contents.
    conn = sqlite3.connect(db_path_for_test)
    try:
        db_df = pd.read_sql_query("SELECT * FROM inventory_parts", conn)
        assert len(db_df) == len(importer.inventory_parts_df)
    finally:
        conn.close()