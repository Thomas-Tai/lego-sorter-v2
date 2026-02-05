"""
Camera Diagnostic Script
Isolates camera capture from inference logic.
Tests:
1. Opening camera with CAP_DSHOW (better for Windows)
2. Warming up (reading 30 frames) to allow Auto-Exposure to settle
3. Saving the final frame for inspection
"""

import cv2
import time
import os

OUTPUT_FILE = r"C:\Users\sky\.gemini\antigravity\brain\95db7288-0bbb-4b60-9e3f-17cbeb125dc2\diagnostic_capture.jpg"


def main():
    print("--- Camera Diagnostic ---")

    # Try index 0 and 1
    for idx in [0, 1]:
        print(f"\nTesting Camera Index {idx}...")

        # On Windows, cv2.CAP_DSHOW (700) is often more reliable than MSMF
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print(f"  Failed to open camera {idx}")
            continue

        # Set Resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Warmup loop - extensive
        print("  Warming up (30 frames)...")
        block_count = 0
        for i in range(30):
            ret, frame = cap.read()
            if not ret:
                print("    Failed to read frame")
                continue

            # Check if frame is purely black
            if frame.sum() == 0:
                block_count += 1
            else:
                # Found non-black frame
                pass

        if block_count == 30:
            print("  WARNING: All 30 warmup frames were pure black!")
        else:
            print(f"  Warmup complete. Black frames: {block_count}/30")

        # Final Capture
        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            filename = OUTPUT_FILE.replace(".jpg", f"_idx{idx}.jpg")
            cv2.imwrite(filename, frame)
            print(f"  Saved diagnostic image to: {filename}")

            # If we got a good image, stop testing
            if frame.sum() > 0:
                print("  Success! Non-black image captured.")
                return
        else:
            print("  Final capture failed.")


if __name__ == "__main__":
    main()
