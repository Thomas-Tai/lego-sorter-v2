import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import tools.run_acquirer
    print("DEBUG: dir(tools.run_acquirer):", dir(tools.run_acquirer))
except Exception as e:
    print(f"DEBUG: Import error in test: {e}")

class TestImageAcquirer(unittest.TestCase):
    def test_dummy(self):
        pass

if __name__ == '__main__':
    unittest.main()
