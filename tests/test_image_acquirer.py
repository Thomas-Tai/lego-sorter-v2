import pytest
import sqlite3
from pathlib import Path
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


@pytest.fixture
def acquirer_instance(test_db: Path, tmp_path: Path) -> ImageAcquirer:
    """A fixture that provides a ready-to-use ImageAcquirer instance."""
    return ImageAcquirer(db_path=str(test_db), output_path=str(tmp_path))


def test_get_parts_to_shoot_returns_only_unshot_parts(acquirer_instance: ImageAcquirer):
    """
    Tests if the _get_parts_to_shoot method only returns parts where
    image_folder_name is NULL.
    """
    # Act
    parts_to_shoot = acquirer_instance._get_parts_to_shoot()

    # Assert
    assert len(parts_to_shoot) == 2
    part_nums = [part[0] for part in parts_to_shoot]
    assert "3001" not in part_nums
    assert "3002" in part_nums
    assert "3003" in part_nums


def test_create_image_directory_creates_folder_and_returns_path(
    acquirer_instance: ImageAcquirer, tmp_path: Path
):
    """
    Tests if _create_image_directory method can:
    1. Create a folder named after the part_num inside the base path.
    2. Return the correct absolute path to the created folder.
    """
    # Arrange
    part_num = "3001"
    expected_path = tmp_path / part_num
    assert not expected_path.exists()

    # Act
    created_path = acquirer_instance._create_image_directory(part_num)

    # Assert
    assert expected_path.exists()
    assert expected_path.is_dir()
    assert created_path == expected_path
