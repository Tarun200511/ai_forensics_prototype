"""
face_recognition_module.py — Suspect Face Recognition Module
AI Edge Forensics Prototype

Detects faces in evidence images and matches them against a
known suspects database using the face_recognition library (dlib-based).
"""

import os
import cv2
import numpy as np
from datetime import datetime

# Try importing face_recognition (requires dlib)
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("[Face] WARNING: face_recognition not installed. Using mock mode.")


# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

MODULE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.dirname(MODULE_DIR)

# Suspects database directory — store one image per known suspect
SUSPECTS_DIR = os.path.join(PROJECT_DIR, "suspects")

# Face matching tolerance: lower = stricter (0.6 is library default)
TOLERANCE = 0.55

# Minimum detection confidence (HOG model doesn't give confidence natively)
# We'll use distance as proxy: distance < TOLERANCE → match
DISTANCE_CONFIDENCE_SCALE = 1.0  # Used to convert distance to % confidence


# ---------------------------------------------------------------
# Suspects Database (in-memory cache)
# ---------------------------------------------------------------

# Format: { "Name": [128-d encoding, ...] }
_suspects_db: dict = {}
_db_loaded: bool = False


def load_suspects_database(suspects_dir: str = None) -> bool:
    """
    Load face encodings for all suspects from the suspects directory.
    Each image file should be named: <Suspect_Name>.jpg

    Args:
        suspects_dir (str): Path to suspect images folder.

    Returns:
        bool: True if at least one suspect was loaded.
    """
    global _suspects_db, _db_loaded

    if suspects_dir is None:
        suspects_dir = SUSPECTS_DIR

    os.makedirs(suspects_dir, exist_ok=True)
    _suspects_db = {}

    if not FACE_RECOGNITION_AVAILABLE:
        # Populate mock data for demo
        _suspects_db["John_Doe (MOCK)"] = []
        _suspects_db["Jane_Smith (MOCK)"] = []
        _db_loaded = True
        return True

    supported = (".jpg", ".jpeg", ".png")
    loaded = 0

    for filename in os.listdir(suspects_dir):
        if not filename.lower().endswith(supported):
            continue

        name = os.path.splitext(filename)[0].replace("_", " ")
        path = os.path.join(suspects_dir, filename)

        try:
            # Load image with face_recognition (RGB)
            img = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(img)

            if encodings:
                _suspects_db[name] = encodings[0]  # Use first face found
                print(f"[Face] Suspect loaded: {name}")
                loaded += 1
            else:
                print(f"[Face] WARNING: No face found in {filename}, skipping.")

        except Exception as e:
            print(f"[Face] ERROR loading {filename}: {e}")

    _db_loaded = True
    print(f"[Face] Suspects database ready: {loaded} suspects.")
    return loaded > 0


def get_suspects_db() -> dict:
    """Return suspects DB, loading it if not already loaded."""
    global _db_loaded
    if not _db_loaded:
        load_suspects_database()
    return _suspects_db


def recognize_faces(image_path: str, output_dir: str = None) -> dict:
    """
    Detect and identify faces in an evidence image.

    Process:
        1. Load image and convert to RGB
        2. Detect face locations (HOG model)
        3. Extract 128-d face encodings
        4. Compare each encoding against suspects DB
        5. Draw bounding boxes and labels
        6. Save annotated image

    Args:
        image_path (str): Path to input evidence image.
        output_dir (str): Where to save annotated output.

    Returns:
        dict: {
            'face_detected': 'Yes' | 'No',
            'face_count': int,
            'matches': [{'name': str, 'confidence': float, 'bbox': list}],
            'face_match': 'Yes' | 'No',
            'face_names': str (comma-separated),
            'annotated_image': str,
            'summary': str
        }
    """
    result = {
        "face_detected": "No",
        "face_count": 0,
        "matches": [],
        "face_match": "No",
        "face_names": "Unknown",
        "annotated_image": image_path,
        "summary": "No faces detected."
    }

    if not os.path.exists(image_path):
        result["summary"] = f"ERROR: File not found: {image_path}"
        return result

    if not FACE_RECOGNITION_AVAILABLE:
        return _mock_face_recognition(image_path, output_dir, result)

    # --- Load image ---
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        result["summary"] = "ERROR: Could not load image."
        return result

    # face_recognition uses RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    annotated = bgr_image.copy()

    # --- Detect face locations ---
    # model='hog' is CPU-friendly; use 'cnn' for GPU (slower start)
    try:
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
    except Exception as e:
        result["summary"] = f"Face detection error: {e}"
        return result

    if not face_locations:
        annotated_path = _save_annotated(annotated, image_path, output_dir, "face")
        result["annotated_image"] = annotated_path
        return result

    result["face_detected"] = "Yes"
    result["face_count"] = len(face_locations)

    # --- Extract face encodings ---
    try:
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    except Exception as e:
        result["summary"] = f"Encoding error: {e}"
        return result

    # --- Load suspects ---
    suspects_db = get_suspects_db()
    known_names = list(suspects_db.keys())
    known_encodings = [suspects_db[n] for n in known_names
                       if not isinstance(suspects_db[n], list)]

    matches_found = []
    identified_names = []

    for i, (enc, loc) in enumerate(zip(face_encodings, face_locations)):
        top, right, bottom, left = loc
        name = "Unknown"
        confidence = 0.0

        if known_encodings:
            # Compare face against all known suspects
            distances = face_recognition.face_distance(known_encodings, enc)
            best_idx = np.argmin(distances)
            best_distance = distances[best_idx]

            if best_distance <= TOLERANCE:
                name = known_names[best_idx]
                # Convert distance to confidence % (distance 0 = 100%, 0.6 = 0%)
                confidence = round((1.0 - best_distance / TOLERANCE) * 100, 1)
                confidence = max(0.0, min(confidence, 100.0))
                identified_names.append(name)

        matches_found.append({
            "face_id": i + 1,
            "name": name,
            "confidence": confidence,
            "bbox": [left, top, right, bottom],
            "status": "IDENTIFIED" if name != "Unknown" else "UNIDENTIFIED"
        })

        # Draw bounding box
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 3)

        # Label background
        label = f"{name} ({confidence:.0f}%)" if name != "Unknown" else "Unknown"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (left, bottom), (left + lw + 6, bottom + lh + 10), color, -1)
        cv2.putText(annotated, label, (left + 3, bottom + lh + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # --- Compile results ---
    result["matches"] = matches_found
    if identified_names:
        result["face_match"] = "Yes"
        result["face_names"] = ", ".join(set(identified_names))
        result["summary"] = (
            f"{len(face_locations)} face(s) detected. "
            f"Identified: {result['face_names']}."
        )
    else:
        result["face_names"] = "No suspects identified"
        result["summary"] = (
            f"{len(face_locations)} face(s) detected. No suspects matched."
        )

    annotated_path = _save_annotated(annotated, image_path, output_dir, "face")
    result["annotated_image"] = annotated_path
    return result


def detect_faces_only(image_path: str) -> list:
    """
    Return only bounding box locations of detected faces (no recognition).
    Lightweight function for quick UI preview.

    Args:
        image_path (str): Path to input image.

    Returns:
        list: List of (top, right, bottom, left) tuples.
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return [(50, 350, 250, 150)]  # Mock bbox

    img = face_recognition.load_image_file(image_path)
    return face_recognition.face_locations(img, model="hog")


def add_suspect(name: str, image_path: str) -> bool:
    """
    Add a new suspect to the in-memory database and save image to suspects folder.

    Args:
        name (str): Suspect's full name (used as filename).
        image_path (str): Path to the suspect's reference photo.

    Returns:
        bool: True if successfully added.
    """
    if not FACE_RECOGNITION_AVAILABLE:
        print("[Face] face_recognition not available — cannot add suspect.")
        return False

    try:
        img = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(img)

        if not encodings:
            print(f"[Face] No face found in image for suspect: {name}")
            return False

        # Save a copy to suspects directory
        os.makedirs(SUSPECTS_DIR, exist_ok=True)
        safe_name = name.replace(" ", "_")
        dest = os.path.join(SUSPECTS_DIR, f"{safe_name}.jpg")
        import shutil
        shutil.copy2(image_path, dest)

        # Update in-memory cache
        _suspects_db[name] = encodings[0]
        print(f"[Face] Suspect added: {name}")
        return True

    except Exception as e:
        print(f"[Face] ERROR adding suspect {name}: {e}")
        return False


# ---------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------

def _save_annotated(image: np.ndarray, original_path: str,
                    output_dir: str, prefix: str) -> str:
    """Save annotated output image and return path."""
    if output_dir is None:
        output_dir = os.path.dirname(original_path)
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"{prefix}_annotated_{ts}.jpg")
    cv2.imwrite(path, image)
    return path


def _mock_face_recognition(image_path: str, output_dir: str, result: dict) -> dict:
    """
    Return mock face recognition result for demo/development.
    Used when face_recognition library is not available.
    """
    print("[Face] Using MOCK face recognition (library not available).")
    image = cv2.imread(image_path)
    if image is None:
        return result

    # Draw a mock bounding box
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 3
    cv2.rectangle(image, (cx - 60, cy - 80), (cx + 60, cy + 80), (0, 255, 0), 3)
    cv2.putText(image, "John Doe (DEMO 87%)", (cx - 80, cy + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    annotated_path = _save_annotated(image, image_path, output_dir, "face_mock")

    result["face_detected"] = "Yes"
    result["face_count"] = 1
    result["face_match"] = "Yes"
    result["face_names"] = "John Doe (MOCK)"
    result["matches"] = [{
        "face_id": 1,
        "name": "John Doe (MOCK)",
        "confidence": 87.0,
        "bbox": [cx - 60, cy - 80, cx + 60, cy + 80],
        "status": "IDENTIFIED"
    }]
    result["annotated_image"] = annotated_path
    result["summary"] = "MOCK: 1 face detected. Identified: John Doe (87% confidence)."
    return result
