# In tests/mocks/mock_vision_service.py

from sorter_app.services.base_service import AbstractVisionService


class MockVisionService(AbstractVisionService):
    """
    A mock implementation of the Vision Service for testing purposes.

    This class does not interact with any real camera. Instead, it records
    the calls made to its methods, allowing tests to assert that high-level
    logic is correctly interacting with the vision layer.
    """

    def __init__(self) -> None:
        self.capture_count = 0
        self.captured_filepaths: list[str] = []
        self.release_called = False
        self.should_succeed = True  # Set to False to simulate capture failures

    def capture_image(self, filepath: str) -> bool:
        """Records a capture attempt and returns success based on should_succeed."""
        self.capture_count += 1
        self.captured_filepaths.append(filepath)
        return self.should_succeed

    def release(self) -> None:
        """Records that release was called."""
        self.release_called = True
