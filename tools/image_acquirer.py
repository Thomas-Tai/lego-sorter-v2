import sqlite3
from typing import List, Tuple


class ImageAcquirer:
    """A tool for automatically acquiring images of Lego parts.

    Args:
        db_path (str): The path to the Lego parts SQLite database.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_parts_to_shoot(self) -> List[Tuple[str, str]]:
        """Queries the database for all parts that have not yet had their images taken.

        This method queries the 'parts' table to find all records where the
        'image_folder_name' column is NULL.

        Returns:
            List[Tuple[str, str]]: A list of tuples, where each tuple contains
                                  (part_num, name).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT part_num, name FROM parts
            WHERE image_folder_name IS NULL
        """
        )
        parts = cursor.fetchall()
        conn.close()
        return parts
