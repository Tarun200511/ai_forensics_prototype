"""
generate_test_data.py — Test Asset Generator
AI Edge Forensics Prototype

Run this script ONCE to generate synthetic test images for:
  - Blood stain patterns (red on dark background)
  - Footprint silhouettes (grayscale shoe outline)
  - Suspect placeholder photos (face-colored rectangles)

Usage:
    python generate_test_data.py
"""

import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def make_blood_stain_image():
    """
    Create a synthetic blood stain pattern image.
    Generates multiple red circular and elongated blobs on a dark background.
    """
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    img[:] = (20, 15, 10)   # Very dark brownish background

    # Impact droplets (small circles)
    for _ in range(40):
        cx = np.random.randint(50, 550)
        cy = np.random.randint(50, 550)
        r = np.random.randint(3, 18)
        darkness = np.random.randint(150, 220)
        cv2.circle(img, (cx, cy), r, (0, 0, darkness), -1)
        cv2.circle(img, (cx, cy), r, (0, 0, 255), 1)

    # Cast-off trail (elongated ellipses)
    for i in range(8):
        cx = 100 + i * 55
        cy = 300 + i * 10
        cv2.ellipse(img, (cx, cy), (25, 6), 30, 0, 360, (0, 0, 200), -1)

    # Drip drops (large circles)
    for _ in range(5):
        cx = np.random.randint(150, 450)
        cy = np.random.randint(150, 450)
        cv2.circle(img, (cx, cy), np.random.randint(20, 45), (0, 0, 180), -1)

    # Slight blur for realism
    img = cv2.GaussianBlur(img, (3, 3), 0)

    save_path = os.path.join(BASE_DIR, "static", "uploads", "test_blood_stain.jpg")
    cv2.imwrite(save_path, img)
    print(f"[TestData] Created: {save_path}")

    # Also create as footprint reference
    fp_dir = os.path.join(BASE_DIR, "static", "footprints")
    os.makedirs(fp_dir, exist_ok=True)


def make_footprint_images():
    """
    Create synthetic shoe print images for the reference database and a test query.
    """
    fp_dir = os.path.join(BASE_DIR, "static", "footprints")
    os.makedirs(fp_dir, exist_ok=True)

    def draw_footprint(img, cx, cy, scale=1.0, angle=0):
        """Draw a simple shoe sole outline."""
        # Sole outline (ellipse)
        cv2.ellipse(img, (cx, cy), (int(50*scale), int(120*scale)),
                    angle, 0, 360, (200, 200, 200), 3)
        # Heel
        cv2.ellipse(img, (cx, cy + int(80*scale)), (int(35*scale), int(30*scale)),
                    angle, 0, 360, (200, 200, 200), 3)
        # Toe box
        cv2.ellipse(img, (cx, cy - int(80*scale)), (int(45*scale), int(35*scale)),
                    angle, 0, 360, (200, 200, 200), 3)
        # Tread lines
        for i in range(-3, 4):
            y = cy + i * int(20*scale)
            cv2.line(img, (cx - int(40*scale), y), (cx + int(40*scale), y),
                     (160, 160, 160), 1)

    # Reference prints
    for name, scale in [("nike_size10", 1.0), ("adidas_size9", 0.9),
                         ("boot_size11", 1.1)]:
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        noise = np.random.randint(0, 15, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        draw_footprint(img, 256, 256, scale=scale)
        path = os.path.join(fp_dir, f"{name}.jpg")
        cv2.imwrite(path, img)
        print(f"[TestData] Footprint reference: {path}")

    # Query print (similar to nike_size10)
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    draw_footprint(img, 256, 256, scale=0.98)
    # Add dirt/smudge effect
    cv2.ellipse(img, (230, 300), (30, 15), 45, 0, 360, (80, 80, 80), -1)

    query_path = os.path.join(BASE_DIR, "static", "uploads", "test_footprint.jpg")
    cv2.imwrite(query_path, img)
    print(f"[TestData] Footprint query: {query_path}")


def make_suspect_placeholder():
    """
    Create a realistic-looking placeholder suspect image.
    (A colored oval on a neutral background to represent a face photograph)
    """
    suspects_dir = os.path.join(BASE_DIR, "suspects")
    os.makedirs(suspects_dir, exist_ok=True)

    for name, skin_color, fname in [
        ("John_Doe",   (130, 150, 200), "John_Doe.jpg"),
        ("Jane_Smith", (140, 160, 210), "Jane_Smith.jpg"),
    ]:
        img = np.full((400, 300, 3), (50, 50, 60), dtype=np.uint8)

        # Face oval
        cv2.ellipse(img, (150, 160), (80, 100), 0, 0, 360, skin_color, -1)
        # Eyes
        cv2.circle(img, (120, 140), 12, (30, 30, 30), -1)
        cv2.circle(img, (180, 140), 12, (30, 30, 30), -1)
        cv2.circle(img, (124, 136), 4, (200, 200, 200), -1)
        cv2.circle(img, (184, 136), 4, (200, 200, 200), -1)
        # Nose
        cv2.ellipse(img, (150, 175), (10, 7), 0, 0, 360, (100, 120, 160), -1)
        # Mouth
        cv2.ellipse(img, (150, 200), (25, 10), 0, 0, 180, (80, 60, 80), 2)
        # Hair
        cv2.ellipse(img, (150, 70), (82, 55), 0, 0, 360, (40, 30, 20), -1)
        # Collar
        cv2.rectangle(img, (80, 300), (220, 400), (60, 60, 80), -1)

        # Name label
        cv2.putText(img, f"SUSPECT: {name}", (10, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        path = os.path.join(suspects_dir, fname)
        cv2.imwrite(path, img)
        print(f"[TestData] Suspect placeholder: {path}")


def make_scene_image():
    """Create a generic dark scene test image."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (15, 20, 25)

    # Some objects
    cv2.rectangle(img, (100, 100), (250, 300), (50, 60, 80), -1)
    cv2.rectangle(img, (350, 150), (550, 380), (60, 50, 70), -1)
    cv2.putText(img, "CRIME SCENE - TEST IMAGE", (50, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)

    path = os.path.join(BASE_DIR, "static", "uploads", "test_scene.jpg")
    cv2.imwrite(path, img)
    print(f"[TestData] Scene image: {path}")


if __name__ == "__main__":
    print("=" * 50)
    print("AI Edge Forensics — Test Data Generator")
    print("=" * 50)
    make_blood_stain_image()
    make_footprint_images()
    make_suspect_placeholder()
    make_scene_image()
    print("\n✅ All test assets created successfully!")
    print("\nTest images are in:")
    print("  • static/uploads/test_blood_stain.jpg")
    print("  • static/uploads/test_footprint.jpg")
    print("  • static/uploads/test_scene.jpg")
    print("  • static/footprints/ (reference database)")
    print("  • suspects/  (placeholder suspect photos)")
