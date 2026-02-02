import os
import cv2
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor

# Configuration
IMAGE_DIR = r"C:\D\WorkSpace\[Local]_Station\01_Heavy_Assets\lego_inventory_parts_ID_Color"
DB_PATH = r"C:\D\WorkSpace\[Cloud]_Company_Sync\MSC\OwnInfo\MyResearchProject\Lego Sorter\_2 labelling_software\project_root_demo\2_lego_database\rebrickableToSqlite3\lego_R_20240627.db"


def check_image(filename):
    filepath = os.path.join(IMAGE_DIR, filename)

    # Check 1: File exists (sanity check)
    if not os.path.exists(filepath):
        return "missing"

    # Check 2: File size > 0
    if os.path.getsize(filepath) == 0:
        return "zero_byte"

    # Check 3: Valid image header (can open with cv2)
    try:
        img = cv2.imread(filepath)
        if img is None:
            return "corrupt"
    except Exception:
        return "corrupt"

    return "valid"


def main():
    print(f"Verifying images in: {IMAGE_DIR}")

    # 1. Get expected images from DB
    print("Querying database for expected images...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT part_num, color_id, img_url FROM inventory_parts WHERE img_url IS NOT NULL AND img_url != ''"
    )
    rows = cursor.fetchall()
    conn.close()

    total_expected = len(rows)
    print(f"Total expected unique images from DB: {total_expected}")

    # 2. Get actual files
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Directory not found: {IMAGE_DIR}")
        return

    actual_files = os.listdir(IMAGE_DIR)
    actual_files_set = set(actual_files)
    print(f"Total files in directory: {len(actual_files)}")

    # 3. Match DB records to files
    print("Checking against database records...")
    missing_files = []

    files_to_check = []

    for row in rows:
        part_num, color_id, url = row
        safe_part_num = str(part_num).replace("/", "_").replace("\\", "_")
        expected_filename = f"{safe_part_num}_{color_id}.jpg"

        if expected_filename not in actual_files_set:
            missing_files.append(expected_filename)
        else:
            files_to_check.append(expected_filename)

    print(f"Missing from DB list: {len(missing_files)}")

    # 4. Check integrity of existing files
    print(f"Checking integrity of {len(files_to_check)} files (Size > 0 & Valid Header)...")

    corrupt_files = []
    zero_byte_files = []

    with ThreadPoolExecutor(max_workers=32) as executor:
        # Submit tasks
        future_to_file = {executor.submit(check_image, f): f for f in files_to_check}

        processed = 0
        total_check = len(files_to_check)

        for future in future_to_file:
            filename = future_to_file[future]
            try:
                result = future.result()
                processed += 1

                if result == "corrupt":
                    corrupt_files.append(filename)
                elif result == "zero_byte":
                    zero_byte_files.append(filename)

                if processed % 1000 == 0:
                    sys.stdout.write(
                        f"\rChecked: {processed}/{total_check} ({(processed/total_check)*100:.1f}%) | Corrupt: {len(corrupt_files)} | 0-Byte: {len(zero_byte_files)}"
                    )
                    sys.stdout.flush()
            except Exception as e:
                print(f"Error checking {filename}: {e}")

    print("\n\n--- Integrity Report ---")
    print(f"Total Expected: {total_expected}")
    print(f"Total Found: {len(actual_files)}")
    print(f"Missing: {len(missing_files)} ({(len(missing_files)/total_expected)*100:.2f}%)")
    print(f"Corrupt (Cannot Open): {len(corrupt_files)}")
    print(f"Zero Byte Files: {len(zero_byte_files)}")

    valid_count = total_expected - len(missing_files) - len(corrupt_files) - len(zero_byte_files)
    print(f"\nFinal Valid Dataset Size: {valid_count} ({(valid_count/total_expected)*100:.2f}%)")


if __name__ == "__main__":
    main()
