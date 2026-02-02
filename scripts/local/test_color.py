"""Quick test for color distinction."""

import requests

API_URL = "http://localhost:8000/v1/predict"

test_images = [
    (
        "3010_25_0.jpg",
        "c:/D/WorkSpace/[Cloud]_Company_Sync/MSC/OwnInfo/MyResearchProject/Lego_Sorter_V2/CodeBase/lego-sorter-v2/data/images/raw/3010/25/3010_25_0.jpg",
    ),
    (
        "3713_4_0.jpg",
        "c:/D/WorkSpace/[Cloud]_Company_Sync/MSC/OwnInfo/MyResearchProject/Lego_Sorter_V2/CodeBase/lego-sorter-v2/data/images/raw/3713/4/3713_4_0.jpg",
    ),
    (
        "3004_1_0.jpg",
        "c:/D/WorkSpace/[Cloud]_Company_Sync/MSC/OwnInfo/MyResearchProject/Lego_Sorter_V2/CodeBase/lego-sorter-v2/data/images/raw/3004/1/3004_1_0.jpg",
    ),
]

print("Testing color distinction...")
print("-" * 60)

for name, path in test_images:
    try:
        with open(path, "rb") as f:
            r = requests.post(API_URL, files={"image": f}, timeout=60)
        m = r.json()["matches"][0]
        print(f"{name:20} -> part={m['part_id']:8} color={m['color_id']:4} conf={m['confidence']:.4f}")
    except Exception as e:
        print(f"{name:20} -> ERROR: {e}")

print("-" * 60)
print("Done!")
