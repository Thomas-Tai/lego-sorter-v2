# tests/test_data_importer_integration.py
# import time
# time.sleep(1) # Pauses for 1 second

import pytest
from pathlib import Path
from tools.data_importer import DataImporter


# Mark this test as an 'integration' test. This allows us to run it separately.
@pytest.mark.integration
def test_importer_with_real_rebrickable_data():
    """
    An integration test to verify that the DataImporter can successfully
    load and filter the real, full dataset downloaded from Rebrickable.
    """
    # --- 1. Arrange ---
    # Point to the real data directory. This test will only run if the data exists.
    real_data_path = Path("data/temp_real_data")
    if not real_data_path.exists():
        pytest.skip("Skipping integration test: Real data directory not found.")

    db_path = real_data_path.parent / "integration_test_output.sqlite"

    importer = DataImporter(raw_data_path=str(real_data_path), db_path=str(db_path))
    importer.target_set_nums = ["45345-1"]  # Spike Essential

    # --- 2. Act ---
    # Execute the core methods we want to verify.
    # We wrap this in a try-except block to provide a more helpful error message.
    try:
        importer._load_csv_files()
        importer._filter_data()
        importer._create_database()
    except Exception as e:
        pytest.fail(f"DataImporter failed to process real data. Error: {e}")

    # --- 3. Assert ---
    # We don't need to check the exact number of parts, but we can verify
    # our core assumptions about the data structure and filtering outcome.

    # Assumption 1: Is the database file available
    assert db_path.exists(), "Database file should be created after running with real data."

    # Assumption 2: Did we load the correct DataFrames?
    assert importer.inventory_parts_df is not None, "inventory_parts_df should be loaded."

    # Assumption 3: Does the loaded DataFrame contain the columns we need?
    required_cols = ["inventory_id", "part_num", "color_id", "quantity"]
    assert all(
        col in importer.inventory_parts_df.columns for col in required_cols
    ), f"Real inventory_parts.csv is missing required columns. Found: {importer.inventory_parts_df.columns.tolist()}"

    # Assumption 4: After filtering, did we get a plausible number of parts?
    # We know Spike Essential has parts, so the result should not be empty.
    assert not importer.inventory_parts_df.empty, "Filtered inventory should not be empty for Spike Essential."

    # Assumption 5: Does the filtered data truly only contain our target inventory?
    # This is the most crucial test of our filtering logic.
    target_inventory_id = 109216  # We can find this ID by inspecting the real CSVs
    assert importer.inventory_parts_df["inventory_id"].unique().tolist() == [
        target_inventory_id
    ], "Filtering logic failed; parts from other sets are still present."
