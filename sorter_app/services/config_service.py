import os
from typing import Optional


class ConfigService:
    """
    Configuration service for the Sorter App.
    Loads settings from environment variables or defaults.
    """

    # Defaults
    DEFAULT_API_URL = "http://localhost:8000"
    DEFAULT_CAMERA_INDEX = 0

    @property
    def api_url(self) -> str:
        """Get the Inference API Base URL."""
        return os.environ.get("LEGO_API_URL", self.DEFAULT_API_URL)

    @property
    def camera_index(self) -> int:
        """Get the camera device index."""
        try:
            return int(
                os.environ.get("LEGO_CAMERA_INDEX", str(self.DEFAULT_CAMERA_INDEX))
            )
        except ValueError:
            return self.DEFAULT_CAMERA_INDEX

    @property
    def debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return os.environ.get("DEBUG", "false").lower() == "true"
