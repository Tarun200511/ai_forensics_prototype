"""
footprint_analysis.py — Forensic Footprint Matching Module
AI Edge Forensics Prototype

Matches an input footprint image against a stored database of
reference footprints using ORB keypoint descriptors and BFMatcher.
"""

import cv2
import numpy as np
import os
import pickle
from datetime import datetime


# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

# Path to the suspects footprint database file (pickled descriptors)
MODULE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.dirname(MODULE_DIR)
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "footprint_matcher.pkl")

# Directory to store reference footprint images
FOOTPRINTS_DIR = os.path.join(PROJECT_DIR, "static", "footprints")

# Feature detector: ORB (no license issues, fast and effective)
ORB_FEATURES = 500          # Number of keypoints to detect
MATCH_RATIO_TEST = 0.75     # Lowe's ratio test threshold (lower = stricter)
MIN_GOOD_MATCHES = 8        # Minimum good matches to consider a hit


def preprocess_footprint(image_path: str) -> np.ndarray:
    """
    Preprocess footprint image for feature extraction.

    Steps:
        1. Load as grayscale
        2. Apply CLAHE for contrast enhancement
        3. Gaussian blur for noise reduction
        4. Canny edge detection

    Args:
        image_path (str): Path to the footprint image file.

    Returns:
        np.ndarray: Preprocessed grayscale image, or None on error.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[Footprint] ERROR: Cannot read {image_path}")
        return None

    # Resize to standard size for consistent matching
    img = cv2.resize(img, (512, 512))

    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Noise reduction
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Edge detection → enhances footprint ridges and outlines
    edges = cv2.Canny(img, 50, 150)

    return edges


def extract_orb_features(image: np.ndarray):
    """
    Extract ORB keypoints and descriptors from a preprocessed image.

    Args:
        image (np.ndarray): Preprocessed grayscale image.

    Returns:
        tuple: (keypoints, descriptors) or ([], None) if no features found.
    """
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    keypoints, descriptors = orb.detectAndCompute(image, None)

    if descriptors is None or len(keypoints) < 5:
        print("[Footprint] WARNING: Too few keypoints found.")
        return [], None

    return keypoints, descriptors


def build_footprint_database(footprints_dir: str = None) -> dict:
    """
    Build a descriptor database from reference footprint images.
    Saves the database as a pickle file for future runs.

    Args:
        footprints_dir (str): Directory containing reference footprint images.

    Returns:
        dict: {filename_without_ext: descriptors (np.ndarray)}
    """
    if footprints_dir is None:
        footprints_dir = FOOTPRINTS_DIR

    os.makedirs(footprints_dir, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    database = {}
    supported_ext = (".jpg", ".jpeg", ".png", ".bmp")

    image_files = [
        f for f in os.listdir(footprints_dir)
        if f.lower().endswith(supported_ext)
    ]

    if not image_files:
        print(f"[Footprint] No reference images found in {footprints_dir}")
        return database

    for filename in image_files:
        path = os.path.join(footprints_dir, filename)
        processed = preprocess_footprint(path)

        if processed is None:
            continue

        _, descriptors = extract_orb_features(processed)

        if descriptors is not None:
            key = os.path.splitext(filename)[0]
            database[key] = descriptors
            print(f"[Footprint] Indexed: {key} ({len(descriptors)} descriptors)")

    # Save database to disk
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(database, f)

    print(f"[Footprint] Database saved: {len(database)} prints → {MODEL_PATH}")
    return database


def load_footprint_database() -> dict:
    """
    Load the footprint descriptor database from disk.
    If no database exists, attempts to build one from /static/footprints/.

    Returns:
        dict: {name: descriptors} or empty dict.
    """
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                db = pickle.load(f)
            print(f"[Footprint] Loaded database: {len(db)} entries.")
            return db
        except Exception as e:
            print(f"[Footprint] Failed to load database: {e}")

    print("[Footprint] No database found. Building from scratch...")
    return build_footprint_database()


def match_footprint(query_image_path: str, output_dir: str = None) -> dict:
    """
    Match an input footprint image against the stored reference database.

    Args:
        query_image_path (str): Path to the crime scene footprint image.
        output_dir (str): Directory to save the annotated match visualization.

    Returns:
        dict: {
            'footprint_match': 'Yes' | 'No',
            'match_id': str (best matching footprint name),
            'match_score': float (0.0 – 100.0 %),
            'annotated_image': str (path to visualization),
            'all_scores': dict {name: score},
            'summary': str
        }
    """
    result = {
        "footprint_match": "No",
        "match_id": "Unknown",
        "match_score": 0.0,
        "annotated_image": query_image_path,
        "all_scores": {},
        "summary": "No footprint match found."
    }

    if not os.path.exists(query_image_path):
        result["summary"] = f"ERROR: File not found: {query_image_path}"
        return result

    # --- Load and preprocess query image ---
    query_processed = preprocess_footprint(query_image_path)
    if query_processed is None:
        result["summary"] = "ERROR: Could not process query image."
        return result

    query_kp, query_desc = extract_orb_features(query_processed)

    if query_desc is None:
        result["summary"] = "Could not extract features from footprint."
        return result

    # --- Load reference database ---
    database = load_footprint_database()

    if not database:
        result["summary"] = "No reference footprints in database."
        return result

    # --- BFMatcher with Hamming distance for ORB ---
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    best_match_name = None
    best_match_score = 0.0
    all_scores = {}

    for ref_name, ref_desc in database.items():
        try:
            # KNN matching with k=2 for ratio test
            matches = bf.knnMatch(query_desc, ref_desc, k=2)

            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < MATCH_RATIO_TEST * n.distance:
                        good_matches.append(m)

            # Score = percentage of good matches relative to possible matches
            max_possible = min(len(query_desc), len(ref_desc))
            score = (len(good_matches) / max_possible * 100) if max_possible > 0 else 0.0
            all_scores[ref_name] = round(score, 2)

            if score > best_match_score:
                best_match_score = score
                best_match_name = ref_name

        except Exception as e:
            print(f"[Footprint] Match error for {ref_name}: {e}")
            all_scores[ref_name] = 0.0

    result["all_scores"] = all_scores

    # --- Determine if match is significant ---
    threshold = 10.0  # Minimum score to call it a match
    if best_match_score >= threshold and best_match_name:
        result["footprint_match"] = "Yes"
        result["match_id"] = best_match_name
        result["match_score"] = round(best_match_score, 2)
        result["summary"] = (
            f"Match found: '{best_match_name}' with {best_match_score:.1f}% similarity."
        )
    else:
        result["summary"] = (
            f"No significant match. Best similarity: {best_match_score:.1f}%"
        )

    # --- Generate visualization ---
    annotated_path = _create_match_visualization(
        query_image_path, query_processed, best_match_name,
        best_match_score, output_dir
    )
    result["annotated_image"] = annotated_path

    return result


def _create_match_visualization(query_path: str, query_processed: np.ndarray,
                                 match_name: str, score: float,
                                 output_dir: str) -> str:
    """
    Create a side-by-side visualization of query vs best match.

    Args:
        query_path: Path to original query image.
        query_processed: Preprocessed query image.
        match_name: Name of the best matching reference print.
        score: Match score percentage.
        output_dir: Where to save output.

    Returns:
        str: Path to saved visualization.
    """
    query_bgr = cv2.imread(query_path)
    if query_bgr is not None:
        query_bgr = cv2.resize(query_bgr, (512, 400))
    else:
        query_bgr = cv2.cvtColor(query_processed, cv2.COLOR_GRAY2BGR)
        query_bgr = cv2.resize(query_bgr, (512, 400))

    # Try to load reference image
    ref_img = None
    if match_name:
        for ext in [".jpg", ".jpeg", ".png"]:
            ref_path = os.path.join(FOOTPRINTS_DIR, match_name + ext)
            if os.path.exists(ref_path):
                ref_img = cv2.imread(ref_path)
                if ref_img is not None:
                    ref_img = cv2.resize(ref_img, (512, 400))
                break

    if ref_img is None:
        ref_img = np.zeros((400, 512, 3), dtype=np.uint8)
        cv2.putText(ref_img, "No DB Image", (150, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 128, 128), 2)

    # Draw divider and labels
    divider = np.zeros((400, 20, 3), dtype=np.uint8)
    canvas = np.hstack([query_bgr, divider, ref_img])

    # Add score banner at top
    banner = np.zeros((60, canvas.shape[1], 3), dtype=np.uint8)
    color = (0, 200, 0) if score >= 20 else (0, 100, 255)
    text = f"Match: {match_name or 'None'}  |  Score: {score:.1f}%"
    cv2.putText(banner, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    final = np.vstack([banner, canvas])

    # Save
    if output_dir is None:
        output_dir = os.path.dirname(query_path)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"footprint_match_{timestamp}.jpg")
    cv2.imwrite(save_path, final)
    return save_path
