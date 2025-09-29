import pytest
import sqlite3
from pathlib import Path

# The ImageAcquirer class is not yet created, so an error here is expected.
from tools.image_acquirer import ImageAcquirer


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    """Creates a temporary database for testing and populates it with data."""
    db_path = tmp_path / "test_lego_parts.sqlite"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the necessary table
    cursor.execute(
        """
        CREATE TABLE parts (
            part_num TEXT PRIMARY KEY,
            name TEXT,
            image_folder_name TEXT
        )
    """
    )

    # Insert test data
    # One part has been shot, two have not.
    cursor.execute("INSERT INTO parts VALUES ('3001', 'Brick 2x4', '3001_folder')")
    cursor.execute("INSERT INTO parts VALUES ('3002', 'Brick 2x3', NULL)")
    cursor.execute("INSERT INTO parts VALUES ('3003', 'Brick 2x2', NULL)")

    conn.commit()
    conn.close()
    return db_path


def test_get_parts_to_shoot_returns_only_unshot_parts(test_db: Path):
    """
    Tests if the _get_parts_to_shoot method only returns parts where
    image_folder_name is NULL.
    """
    # Arrange
    acquirer = ImageAcquirer(db_path=str(test_db))

    # Act
    parts_to_shoot = acquirer._get_parts_to_shoot()

    # Assert
    assert len(parts_to_shoot) == 2
    part_nums = [part[0] for part in parts_to_shoot]
    assert "3001" not in part_nums
    assert "3002" in part_nums
    assert "3003" in part_nums
