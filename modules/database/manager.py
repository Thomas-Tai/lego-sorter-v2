"""
Database Manager Module
Provides DatabaseManager class for LEGO parts database operations.
"""

import sqlite3
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Default database path (relative to project root)
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "db", "lego_parts.sqlite")


@dataclass
class LegoPart:
    part_num: str
    name: str
    color_id: int
    color_name: str
    image_path: Optional[str] = None


class DatabaseManager:
    """Manages SQLite database for LEGO parts."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        self.db_path = os.path.abspath(db_path)
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._create_tables()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _create_tables(self):
        conn = self._get_connection()
        cursor = conn.cursor()

        # Sets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sets (
                set_num TEXT PRIMARY KEY,
                name TEXT,
                year INTEGER,
                theme_id INTEGER,
                num_parts INTEGER
            )
        """)

        # Parts table
        # image_folder_name stores the local path relative to dataset root
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parts (
                part_num TEXT PRIMARY KEY,
                name TEXT,
                img_url TEXT,
                image_folder_name TEXT
            )
        """)

        # Colors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS colors (
                id INTEGER PRIMARY KEY,
                name TEXT,
                rgb TEXT,
                is_trans BOOLEAN
            )
        """)

        # Inventories table (Link between Sets and Inventory Parts)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inventories (
                id INTEGER PRIMARY KEY,
                version INTEGER,
                set_num TEXT,
                FOREIGN KEY (set_num) REFERENCES sets (set_num)
            )
        """)

        # Inventory Parts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inventory_parts (
                inventory_id INTEGER,
                part_num TEXT,
                color_id INTEGER,
                quantity INTEGER,
                is_spare BOOLEAN,
                img_url TEXT,
                FOREIGN KEY (inventory_id) REFERENCES inventories (id),
                FOREIGN KEY (part_num) REFERENCES parts (part_num),
                FOREIGN KEY (color_id) REFERENCES colors (id)
            )
        """)

        conn.commit()
        conn.close()

    def get_parts_in_set(self, set_num: str) -> List[Tuple[str, str, int, str, Optional[str]]]:
        """
        Retrieve all distinct parts for a given set.
        Returns list of (part_num, part_name, color_id, color_name, image_folder_name)
        """
        query = """
            SELECT DISTINCT p.part_num, p.name, c.id, c.name, p.image_folder_name
            FROM sets s
            JOIN inventories i ON s.set_num = i.set_num
            JOIN inventory_parts ip ON i.id = ip.inventory_id
            JOIN parts p ON ip.part_num = p.part_num
            JOIN colors c ON ip.color_id = c.id
            WHERE s.set_num = ? AND ip.is_spare = 0
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(query, (set_num,))
        results = cursor.fetchall()
        conn.close()
        return results

    def update_part_image_folder(self, part_num: str, folder_name: str):
        """Update the image_folder_name for a part (marking it as photographed)"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE parts
            SET image_folder_name = ?
            WHERE part_num = ?
        """,
            (folder_name, part_num),
        )
        conn.commit()
        conn.close()

    def get_unphotographed_parts(self, set_num: str) -> List[Tuple[str, str, int, str]]:
        """Get parts in a set that haven't been photographed yet"""
        all_parts = self.get_parts_in_set(set_num)
        return [p for p in all_parts if p[4] is None]  # p[4] is image_folder_name

    def close(self):
        pass
