"""
Data Importer Module
Provides DataImporter class for importing Rebrickable CSV data.
"""

import csv
import os
import sqlite3
from typing import Dict, List, Any
import logging

# Default paths (relative to project root)
DEFAULT_RAW_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "raw", "rebrickable_20250917"
)


class DataImporter:
    """Imports Rebrickable CSV data into SQLite database."""

    def __init__(self, db_path: str, raw_data_dir: str = None):
        self.db_path = db_path
        if raw_data_dir is None:
            raw_data_dir = DEFAULT_RAW_DATA_DIR
        self.raw_data_dir = os.path.abspath(raw_data_dir)
        self.logger = logging.getLogger(__name__)

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def import_all(self):
        """Import all necessary Rebrickable CSVs"""
        self.import_colors()
        self.import_parts()
        self.import_sets()
        self.import_inventories()
        self.import_inventory_parts()

    def _read_csv(self, filename: str) -> List[Dict[str, Any]]:
        file_path = os.path.join(self.raw_data_dir, filename)
        if not os.path.exists(file_path):
            self.logger.warning(f"File not found: {file_path}")
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def import_colors(self):
        self.logger.info("Importing colors...")
        rows = self._read_csv("colors.csv")
        data = []
        for row in rows:
            is_trans = 1 if row.get("is_trans") in ("t", "True") else 0
            data.append((row["id"], row["name"], row["rgb"], is_trans))

        conn = self._get_connection()
        try:
            conn.executemany(
                "INSERT OR REPLACE INTO colors (id, name, rgb, is_trans) "
                "VALUES (?, ?, ?, ?)",
                data,
            )
            conn.commit()
            self.logger.info(f"Imported {len(data)} colors.")
        except Exception as e:
            self.logger.error(f"Error importing colors: {e}")
        finally:
            conn.close()

    def import_parts(self):
        self.logger.info("Importing parts...")
        file_path = os.path.join(self.raw_data_dir, "parts.csv")
        if not os.path.exists(file_path):
            self.logger.warning("parts.csv not found")
            return

        conn = self._get_connection()
        batch_size = 10000
        batch = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_url = row.get("img_url", None)
                    batch.append(
                        (row["part_num"], row["name"], img_url, None)
                    )  # image_folder_name starts empty

                    if len(batch) >= batch_size:
                        conn.executemany(
                            "INSERT OR IGNORE INTO parts "
                            "(part_num, name, img_url, image_folder_name) "
                            "VALUES (?, ?, ?, ?)",
                            batch,
                        )
                        batch = []

            if batch:
                conn.executemany(
                    "INSERT OR IGNORE INTO parts "
                    "(part_num, name, img_url, image_folder_name) "
                    "VALUES (?, ?, ?, ?)",
                    batch,
                )

            conn.commit()
            self.logger.info("Imported parts.")
        except Exception as e:
            self.logger.error(f"Error importing parts: {e}")
        finally:
            conn.close()

    def import_sets(self):
        self.logger.info("Importing sets...")
        file_path = os.path.join(self.raw_data_dir, "sets.csv")
        if not os.path.exists(file_path):
            return

        conn = self._get_connection()
        batch_size = 5000
        batch = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    batch.append(
                        (
                            row["set_num"],
                            row["name"],
                            int(row["year"]),
                            int(row["theme_id"]),
                            int(row["num_parts"]),
                        )
                    )

                    if len(batch) >= batch_size:
                        conn.executemany(
                            "INSERT OR REPLACE INTO sets "
                            "(set_num, name, year, theme_id, num_parts) "
                            "VALUES (?, ?, ?, ?, ?)",
                            batch,
                        )
                        batch = []

            if batch:
                conn.executemany(
                    "INSERT OR REPLACE INTO sets "
                    "(set_num, name, year, theme_id, num_parts) "
                    "VALUES (?, ?, ?, ?, ?)",
                    batch,
                )
            conn.commit()
            self.logger.info("Imported sets.")
        except Exception as e:
            self.logger.error(f"Error importing sets: {e}")
        finally:
            conn.close()

    def import_inventories(self):
        self.logger.info("Importing inventories...")
        file_path = os.path.join(self.raw_data_dir, "inventories.csv")
        if not os.path.exists(file_path):
            return

        conn = self._get_connection()
        batch = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    batch.append((int(row["id"]), int(row["version"]), row["set_num"]))
                    if len(batch) >= 10000:
                        conn.executemany(
                            "INSERT OR REPLACE INTO inventories "
                            "(id, version, set_num) VALUES (?, ?, ?)",
                            batch,
                        )
                        batch = []
            if batch:
                conn.executemany(
                    "INSERT OR REPLACE INTO inventories "
                    "(id, version, set_num) VALUES (?, ?, ?)",
                    batch,
                )
            conn.commit()
            self.logger.info("Imported inventories.")
        except Exception as e:
            self.logger.error(f"Error importing inventories: {e}")
        finally:
            conn.close()

    def import_inventory_parts(self):
        self.logger.info("Importing inventory_parts (this might take a while)...")
        file_path = os.path.join(self.raw_data_dir, "inventory_parts.csv")
        if not os.path.exists(file_path):
            return

        conn = self._get_connection()
        batch_size = 10000
        batch = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    is_spare = 1 if row.get("is_spare") in ("t", "True") else 0
                    batch.append(
                        (
                            int(row["inventory_id"]),
                            row["part_num"],
                            int(row["color_id"]),
                            int(row["quantity"]),
                            is_spare,
                            row.get("img_url", None),
                        )
                    )

                    if len(batch) >= batch_size:
                        conn.executemany(
                            "INSERT OR IGNORE INTO inventory_parts "
                            "(inventory_id, part_num, color_id, quantity, "
                            "is_spare, img_url) VALUES (?, ?, ?, ?, ?, ?)",
                            batch,
                        )
                        batch = []

            if batch:
                conn.executemany(
                    "INSERT OR IGNORE INTO inventory_parts "
                    "(inventory_id, part_num, color_id, quantity, "
                    "is_spare, img_url) VALUES (?, ?, ?, ?, ?, ?)",
                    batch,
                )
            conn.commit()
            self.logger.info("Imported inventory_parts.")
        except Exception as e:
            self.logger.error(f"Error importing inventory_parts: {e}")
        finally:
            conn.close()
