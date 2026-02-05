"""
Script to test different rembg parameters on a sample image.
Usage: python scripts/local/test_rembg_params.py <path_to_image>
"""

import sys
import os
from rembg import remove, new_session
from PIL import Image
import io


def test_params(input_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Testing rembg params on: {input_path}")

    with open(input_path, "rb") as f:
        img_bytes = f.read()

    # Define experiments
    experiments = [
        ("default", {}),
        ("alpha_matting", {"alpha_matting": True}),
        ("am_bg_10", {"alpha_matting": True, "alpha_matting_background_threshold": 10}),
        (
            "am_bg_240",
            {"alpha_matting": True, "alpha_matting_background_threshold": 240},
        ),  # Try extreme
        (
            "am_fg_240",
            {"alpha_matting": True, "alpha_matting_foreground_threshold": 240},
        ),
        ("post_process", {"post_process_mask": True}),
    ]

    # Initialize session (u2net is default)
    session = new_session("u2net")

    for name, kwargs in experiments:
        print(f"Running: {name}...")
        try:
            output_bytes = remove(img_bytes, session=session, **kwargs)

            # Save output
            out_img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

            # Composite on white for easier viewing
            bg = Image.new("RGBA", out_img.size, (255, 255, 255, 255))
            composite = Image.alpha_composite(bg, out_img)

            save_path = os.path.join(output_dir, f"{name}.jpg")
            composite.convert("RGB").save(save_path, quality=95)
            print(f"  Saved to {save_path}")

        except Exception as e:
            print(f"  Failed {name}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide image path.")
        sys.exit(1)

    img_path = sys.argv[1]
    out_dir = os.path.join(os.path.dirname(img_path), "rembg_test")
    test_params(img_path, out_dir)
