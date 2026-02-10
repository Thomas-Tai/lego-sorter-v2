"""Database service - thin wrapper over modules.database.manager."""

import logging
from typing import List, Optional, Tuple

from modules.database.manager import DatabaseManager, LegoPart

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service layer for database operations.

    Provides a high-level interface to the DatabaseManager,
    following the service pattern used by other sorter_app services.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize the database service.

        Args:
            db_path: Path to the SQLite database file.
                     Uses the default path if not specified.
        """
        self._db = DatabaseManager(db_path)
        logger.info("DatabaseService initialized (db: %s)", self._db.db_path)

    def get_parts_in_set(
        self, set_num: str
    ) -> List[Tuple[str, str, int, str, Optional[str]]]:
        """Retrieve all distinct parts for a given set.

        Args:
            set_num: The LEGO set number to query.

        Returns:
            List of tuples: (part_num, name, color_id, color_name, image_folder).
        """
        return self._db.get_parts_in_set(set_num)

    def get_unphotographed_parts(self, set_num: str) -> List[Tuple[str, str, int, str]]:
        """Get parts in a set that haven't been photographed yet.

        Args:
            set_num: The LEGO set number to query.

        Returns:
            List of tuples for parts with no image folder.
        """
        return self._db.get_unphotographed_parts(set_num)

    def update_part_image_folder(self, part_num: str, folder_name: str) -> None:
        """Mark a part as photographed by setting its image folder.

        Args:
            part_num: The part number to update.
            folder_name: The folder name containing the part's images.
        """
        self._db.update_part_image_folder(part_num, folder_name)

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()
