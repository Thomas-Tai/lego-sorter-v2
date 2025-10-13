import sqlite3
from typing import List, Tuple
from pathlib import Path  # Import the Path object from pathlib


class ImageAcquirer:
    """A tool for automatically acquiring images of Lego parts.

    Args:
        db_path (str): The path to the Lego parts SQLite database.
        output_path (str): The root directory path for storing acquired images.
    """

    def __init__(self, db_path: str, output_path: str):
        self.db_path = db_path
        self.output_path = Path(output_path)  # Convert the string to a Path object
        self.output_path.mkdir(
            parents=True, exist_ok=True
        )  # Ensure the base output directory exists

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

    def _create_image_directory(self, part_num: str) -> Path:
        """Creates a dedicated image storage directory based on the part number.

        This method creates a subdirectory named after the part_num under
        self.output_path. If parent directories do not exist, they will be
        created. If the target directory already exists, it will not raise
        an error.

        Args:
            part_num (str): The part number for which to create the directory.

        Returns:
            Path: A Path object pointing to the created or existing directory.
        """
        part_dir = self.output_path / part_num
        part_dir.mkdir(parents=True, exist_ok=True)
        return part_dir
