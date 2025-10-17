import sqlite3
from typing import List, Tuple
from pathlib import Path  # Import the Path object from pathlib
from sorter_app.services.base_service import AbstractHardwareService


class ImageAcquirer:
    """A tool for automatically acquiring images of Lego parts.

    Args:
        db_path (str): The path to the Lego parts SQLite database.
        output_path (str): The root directory path for storing acquired images.
    """

    def __init__(
        self,
        db_path: str,
        output_path: str,
        hardware_service: AbstractHardwareService,  # <-- ADD THIS ARGUMENT
    ):
        """Initializes the ImageAcquirer.

        Args:
            db_path: Path to the SQLite database.
            output_path: Root directory to save captured images.
            hardware_service: An object that conforms to the AbstractHardwareService interface.
        """
        self.db_path = db_path
        self.output_path = Path(output_path)
        self.hardware = hardware_service

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

    # (In tools/image_acquirer.py)

    def _prompt_user(self, part_info: Tuple[str, str]) -> None:
        """Displays a prompt in the command line and waits for user confirmation.

        This method prints a formatted prompt to standard output, informing the
        operator of the next part number and name to be placed. The program
        will then block until the operator presses the Enter key.

        Args:
            part_info (Tuple[str, str]): A tuple containing (part_num, part_name).
        """
        part_num, part_name = part_info
        prompt_message = (
            "\n==================================================\n"
            f"  Please place part: {part_num} ({part_name})\n"
            "=================================================="
        )
        # Step 1: Explicitly PRINT the message to standard output.
        print(prompt_message)

        # Step 2: Use input() SOLELY for pausing execution.
        input("  Press ENTER to continue...")

    def _capture_single_part_routine(self) -> None:
        """Executes the complete multi-angle image acquisition routine for a single part.

        This method orchestrates the hardware service to perform a standardized
        capture sequence:
        Initialize Hardware -> Turn LEDs On -> Loop[Capture -> Rotate] -> Turn LEDs Off -> Cleanup Hardware.

        Note: This method relies on the injected hardware_service.
        """

        # Follow the contract defined in our tests.
        self.hardware.setup()
        self.hardware.set_led_power(True)

        # Assuming we define 6 steps of 60 degrees each to capture 6 photos.
        num_steps = 6
        degrees_per_step = 360 // num_steps

        for i in range(num_steps):
            print(f"  Capturing angle {i+1}/{num_steps}...")
            # TODO: Call vision service to capture image here

            self.hardware.turn_turntable(degrees_per_step)

        self.hardware.set_led_power(False)
        self.hardware.cleanup()

    def _update_database(self, part_num: str, folder_name: str) -> None:
        """Updates the image folder name for the specified part in the database.

        This method performs an UPDATE operation on the 'parts' table,
        setting the value of the 'image_folder_name' column for the record
        that matches the given 'part_num'. This operation is permanent.

        Args:
            part_num (str): The part number of the record to update.
            folder_name (str): The folder name to write into the column.
        """
        conn = sqlite3.connect(self.db_path)
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
