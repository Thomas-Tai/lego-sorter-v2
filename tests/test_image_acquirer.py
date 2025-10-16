import pytest
import sqlite3
from pathlib import Path
from tools.image_acquirer import ImageAcquirer
from tests.mocks.mock_hardware_service import MockHardwareService


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
def mock_hardware() -> MockHardwareService:
    """A fixture that provides a clean MockHardwareService instance."""
    return MockHardwareService()


@pytest.fixture
def acquirer_instance(
    test_db: Path, tmp_path: Path, mock_hardware: MockHardwareService
) -> ImageAcquirer:
    """A fixture that provides an ImageAcquirer instance with a mocked hardware service injected."""
    return ImageAcquirer(
        db_path=str(test_db),
        output_path=str(tmp_path),
        hardware_service=mock_hardware,  # Pass the practice knife to the chef.
    )


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


def test_prompt_user_displays_message_and_waits_for_input(
    acquirer_instance: ImageAcquirer, capsys, monkeypatch
):
    """
    Tests if the _prompt_user method can:
    1. Display the correct prompt message to standard output.
    2. Pause the program until the user provides input.
    """
    # Arrange
    part_to_prompt = ("3001", "Brick 2x4")
    # We will only check for the most critical part of the message.
    # This makes the test more robust against minor formatting changes.
    expected_key_message = "Please place part: 3001 (Brick 2x4)"

    # Monkeypatch the input() function.
    # When input() is called, it will immediately return, simulating a user press.
    monkeypatch.setattr("builtins.input", lambda _: None)

    # Act
    acquirer_instance._prompt_user(part_to_prompt)

    # Assert
    # Capture the content that was printed to the screen.
    captured = capsys.readouterr()

    # Verify that the captured output contains our key expected message.
    assert expected_key_message in captured.out


def test_capture_single_part_routine_happy_path(
    acquirer_instance: ImageAcquirer, mock_hardware: MockHardwareService
):
    """
    Tests the core workflow of _capture_single_part_routine (happy path).

    Verifies that this method correctly orchestrates the hardware service
    to complete the following steps:
    1. Turn on the light.
    2. Loop 6 times, turning 60 degrees each time.
    3. Turn off the light.
    4. Perform cleanup at the end.
    """
    # Arrange
    # We assume the camera successfully "takes a picture" after each turn.
    # For now, we will not simulate the camera and will focus only on hardware orchestration.

    # Act
    acquirer_instance._capture_single_part_routine()

    # Assert - Check the "flight log"
    # Check the core metrics the "examiner" cares about.
    assert mock_hardware.setup_called is True
    assert mock_hardware.led_is_on is False  # The light is off at the end.
    assert mock_hardware.turntable_turned_by == 360  # Turned a total of 360 degrees.
    assert mock_hardware.cleanup_called is True
