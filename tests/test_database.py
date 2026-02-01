import unittest
import sqlite3
import os
from sorter_app.data.database_manager import DatabaseManager

class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        # Use a temporary file for DB persistence across connections
        self.test_db_file = "test_lego_parts.sqlite"
        if os.path.exists(self.test_db_file):
            os.remove(self.test_db_file)
            
        self.db = DatabaseManager(self.test_db_file)
        
        # Populate with some dummy data for testing
        self._populate_dummy_data()

    def _populate_dummy_data(self):
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        # Insert a Set
        cursor.execute("INSERT INTO sets (set_num, name, year, theme_id, num_parts) VALUES ('1234-1', 'Test Set', 2023, 1, 10)")
        
        # Insert Parts
        cursor.execute("INSERT INTO parts (part_num, name, img_url, image_folder_name) VALUES ('3001', 'Brick 2x4', 'http://img', NULL)")
        cursor.execute("INSERT INTO parts (part_num, name, img_url, image_folder_name) VALUES ('3002', 'Brick 2x3', 'http://img', 'photos/3002')")
        
        # Insert Colors
        cursor.execute("INSERT INTO colors (id, name, rgb, is_trans) VALUES (15, 'White', 'FFFFFF', 0)")
        cursor.execute("INSERT INTO colors (id, name, rgb, is_trans) VALUES (19, 'Tan', 'E4CD9E', 0)")

        # Insert Inventory
        cursor.execute("INSERT INTO inventories (id, version, set_num) VALUES (100, 1, '1234-1')")
        
        # Insert Inventory Parts
        # Part 3001 in White (qty 5)
        cursor.execute("INSERT INTO inventory_parts (inventory_id, part_num, color_id, quantity, is_spare, img_url) VALUES (100, '3001', 15, 5, 0, 'http://img')")
        # Part 3002 in Tan (qty 2)
        cursor.execute("INSERT INTO inventory_parts (inventory_id, part_num, color_id, quantity, is_spare, img_url) VALUES (100, '3002', 19, 2, 0, 'http://img')")

        conn.commit()
        conn.close()

    def tearDown(self):
        self.db.close()
        if os.path.exists(self.test_db_file):
            os.remove(self.test_db_file)

    def test_get_parts_in_set(self):
        parts = self.db.get_parts_in_set('1234-1')
        self.assertEqual(len(parts), 2)
        
        # Check first part (ordering might vary, so convert to set or sort if needed, but list is okay for small data)
        # Expected: ('3001', 'Brick 2x4', 15, 'White', None) or ...
        part_nums = [p[0] for p in parts]
        self.assertIn('3001', part_nums)
        self.assertIn('3002', part_nums)

    def test_get_unphotographed_parts(self):
        parts = self.db.get_unphotographed_parts('1234-1')
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0][0], '3001') # 3001 has NULL image_folder_name in dummy data

    def test_update_part_image_folder(self):
        self.db.update_part_image_folder('3001', 'photos/3001')
        
        parts = self.db.get_unphotographed_parts('1234-1')
        self.assertEqual(len(parts), 0)

        # Verify update persisted
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT image_folder_name FROM parts WHERE part_num='3001'")
        folder = cursor.fetchone()[0]
        self.assertEqual(folder, 'photos/3001')
        conn.close()

if __name__ == '__main__':
    unittest.main()
