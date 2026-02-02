"""A tool for automatically acquiring images of Lego parts."""

import logging
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

from sorter_app.services.base_service import (
    AbstractHardwareService,
    AbstractVisionService,
)

logger = logging.getLogger(__name__)


class ImageAcquirer:
    """A tool for automatically acquiring images of Lego parts.

    This tool orchestrates the hardware (turntable, LEDs) and vision (camera)
    services to capture multi-angle images of LEGO parts for training data.

    Attributes:
        db_path: Path to the Lego parts SQLite database.
        output_path: Root directory path for storing acquired images.
    """

    def __init__(
        self,
        db_path: str,
        output_path: str,
        hardware_service: AbstractHardwareService,
        vision_service: AbstractVisionService,
    ) -> None:
        """Initializes the ImageAcquirer.

        Args:
            db_path: Path to the SQLite database.
            output_path: Root directory to save captured images.
            hardware_service: An object that conforms to AbstractHardwareService.
            vision_service: An object that conforms to AbstractVisionService.
        """
        self._db_path = db_path
        self._output_path = Path(output_path)
        self._hardware = hardware_service
        self._vision = vision_service
        self._current_part_dir: Optional[Path] = None

        self._output_path.mkdir(parents=True, exist_ok=True)
        logger.info("ImageAcquirer initialized. Output path: %s", self._output_path)

    def _get_parts_to_shoot(self) -> List[Tuple[str, str]]:
        """Queries the database for all parts that have not yet had their images taken.

        This method queries the 'parts' table to find all records where the
        'image_folder_name' column is NULL.

        Returns:
            A list of tuples, where each tuple contains (part_num, name).
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT part_num, name FROM parts
            WHERE image_folder_name IS NULL
        """)
        parts = cursor.fetchall()
        conn.close()
        logger.info("Found %d parts to shoot", len(parts))
        return parts

    def _create_image_directory(self, part_num: str) -> Path:
        """Creates a dedicated image storage directory based on the part number.

        This method creates a subdirectory named after the part_num under
        self._output_path. If parent directories do not exist, they will be
        created. If the target directory already exists, it will not raise
        an error.

        Args:
            part_num: The part number for which to create the directory.

        Returns:
            A Path object pointing to the created or existing directory.
        """
        part_dir = self._output_path / part_num
        part_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Created image directory: %s", part_dir)
        return part_dir

    def _prompt_user(self, part_info: Tuple[str, str]) -> None:
        """Displays a prompt in the command line and waits for user confirmation.

        This method prints a formatted prompt to standard output, informing the
        operator of the next part number and name to be placed. The program
        will then block until the operator presses the Enter key.

        Args:
            part_info: A tuple containing (part_num, part_name).
        """
        part_num, part_name = part_info
        prompt_message = (
            "\n==================================================\n"
            f"  Please place part: {part_num} ({part_name})\n"
            "=================================================="
        )
        print(prompt_message)
        input("  Press ENTER to continue...")

    def _capture_single_part_routine(self) -> None:
        """Executes the complete multi-angle image acquisition routine for a single part.

        This method orchestrates the hardware and vision services to perform a
        standardized capture sequence:
        Initialize Hardware -> Turn LEDs On -> Loop[Capture -> Rotate] ->
        Turn LEDs Off -> Cleanup Hardware.

        Requires self._current_part_dir to be set before calling.
        """
        if self._current_part_dir is None:
            logger.error("_current_part_dir is not set. Cannot capture.")
            return

        self._hardware.setup()
        self._hardware.set_led_power(True)

        # Capture 6 angles at 60 degree intervals
        num_steps = 6
        degrees_per_step = 360 // num_steps

        for i in range(num_steps):
            filename = self._current_part_dir / f"angle_{i + 1:02d}.jpg"
            logger.info("Capturing angle %d/%d -> %s", i + 1, num_steps, filename.name)

            if not self._vision.capture_image(str(filename)):
                logger.warning("Failed to capture image at angle %d", i + 1)

            self._hardware.turn_turntable(degrees_per_step)

        self._hardware.set_led_power(False)
        self._hardware.cleanup()
        logger.info("Capture routine complete for %s", self._current_part_dir.name)

    def _update_database(self, part_num: str, folder_name: str) -> None:
        """Updates the image folder name for the specified part in the database.

        This method performs an UPDATE operation on the 'parts' table,
        setting the value of the 'image_folder_name' column for the record
        that matches the given 'part_num'. This operation is permanent.

        Args:
            part_num: The part number of the record to update.
            folder_name: The folder name to write into the column.
        """
        conn = sqlite3.connect(self._db_path)
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
        logger.debug("Database updated: %s -> %s", part_num, folder_name)

    def run(self) -> None:
        """Runs the full image acquisition workflow.

        Iterates through all parts that need images, prompts the user to place
        each part, captures multi-angle images, and updates the database.
        """
        parts = self._get_parts_to_shoot()

        if not parts:
            logger.info("No parts to shoot. All parts have images.")
            return

        logger.info("Starting acquisition for %d parts", len(parts))

        for part_info in parts:
            part_num, _ = part_info

            # Prompt user to place the part
            self._prompt_user(part_info)

            # Create directory for this part's images
            self._current_part_dir = self._create_image_directory(part_num)

            # Run capture routine
            self._capture_single_part_routine()

            # Update database to mark as complete
            self._update_database(part_num, part_num)

        # Release camera when done
        self._vision.release()
        logger.info("Acquisition complete for all parts")


def main() -> None:
    """Main entry point for the image acquirer tool."""
    import logging
    import sys
    from pathlib import Path

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Determine paths
    project_root = Path(__file__).parent.parent
    db_path = project_root / "data" / "processed" / "lego_parts.sqlite"
    output_path = project_root / "data" / "images"

    # Check if database exists
    if not db_path.exists():
        logger.error("Database not found at %s", db_path)
        logger.error("Please run the data importer first: python run_importer.py")
        sys.exit(1)

    # Import and create services
    from sorter_app.services.hardware_service import RaspberryPiHardwareService
    from sorter_app.services.vision_service import RaspberryPiVisionService

    logger.info("Initializing services...")

    try:
        hardware_service = RaspberryPiHardwareService()
        vision_service = RaspberryPiVisionService(camera_index=0)
    except Exception as e:
        logger.error("Failed to initialize services: %s", e)
        sys.exit(1)

    # Create and run the acquirer
    acquirer = ImageAcquirer(
        db_path=str(db_path),
        output_path=str(output_path),
        hardware_service=hardware_service,
        vision_service=vision_service,
    )

    try:
        acquirer.run()
    except KeyboardInterrupt:
        logger.info("Acquisition interrupted by user")
    finally:
        vision_service.release()


if __name__ == "__main__":
    main()
