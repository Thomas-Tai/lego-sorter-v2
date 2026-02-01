import sqlite3
import os
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Configuration
DB_PATH = r"C:\D\WorkSpace\[Cloud]_Company_Sync\MSC\OwnInfo\MyResearchProject\Lego Sorter\_2 labelling_software\project_root_demo\2_lego_database\rebrickableToSqlite3\lego_R_20240627.db"
OUTPUT_DIR = r"C:\D\WorkSpace\[Local]_Station\01_Heavy_Assets\lego_inventory_parts_ID_Color"
MAX_WORKERS = 32  # Increased from 8 to 32 for IO-bound task

def download_image(data, output_folder):
    part_num, color_id, url = data
    if not url:
        return "skipped"
        
    try:
        # Construct filename: PartNum_ColorID.jpg
        # Sanitize part_num to replace invalid characters like / with _
        safe_part_num = str(part_num).replace("/", "_").replace("\\", "_")
        filename = f"{safe_part_num}_{color_id}.jpg"
        file_path = os.path.join(output_folder, filename)

        if os.path.exists(file_path):
            return "exists"

        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return "downloaded"
        else:
            return f"failed_{response.status_code}"
            
    except Exception as e:
        return f"error_{str(e)}"

def main():
    print(f"Connecting to DB: {DB_PATH}")
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("Querying image URLs...")
    # Select DISTINCT part_num, color_id, img_url to handle duplicates and get naming data
    cursor.execute("SELECT DISTINCT part_num, color_id, img_url FROM inventory_parts WHERE img_url IS NOT NULL AND img_url != ''")
    rows = cursor.fetchall()
    conn.close()
    
    total_images = len(rows)
    print(f"Found {total_images} unique images to process.")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    print(f"Starting download with {MAX_WORKERS} threads...")
    
    stats = {"downloaded": 0, "exists": 0, "failed": 0, "skipped": 0}
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks. Pass the whole tuple (part_num, color_id, url)
        futures = {executor.submit(download_image, row, OUTPUT_DIR): row for row in rows}
        
        processed = 0
        for future in as_completed(futures):
            processed += 1
            result = future.result()
            
            if result == "downloaded":
                stats["downloaded"] += 1
            elif result == "exists":
                stats["exists"] += 1
            elif result == "skipped":
                stats["skipped"] += 1
            else:
                stats["failed"] += 1

                
            # Progress update every 100 images
            if processed % 100 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                params = (processed, total_images, processed/total_images*100, rate, stats['downloaded'], stats['exists'])
                sys.stdout.write(f"\rProgress: {processed}/{total_images} ({processed/total_images*100:.1f}%) | Rate: {rate:.1f} img/s | New: {stats['downloaded']} | Exists: {stats['exists']}")
                sys.stdout.flush()
                
    print("\n\nDownload Complete!")
    print(f"Total: {total_images}")
    print(f"Downloaded: {stats['downloaded']}")
    print(f"Already Existed: {stats['exists']}")
    print(f"Failed: {stats['failed']}")

if __name__ == "__main__":
    main()
