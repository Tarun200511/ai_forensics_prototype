"""
weapon_detection.py — YOLOv8 Weapon Detection Module
AI Edge Forensics Prototype

Detects weapons (knife, gun, baseball bat) in images using
the YOLOv8 pretrained COCO model from Ultralytics.
"""

import cv2
import os
import numpy as np
from datetime import datetime

# Try to import ultralytics YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[Weapon] WARNING: ultralytics not installed. Using mock detection.")


# ---------------------------------------------------------------
# COCO classes relevant to forensic weapon detection
# Maps COCO class names to forensic display labels
# ---------------------------------------------------------------
WEAPON_CLASSES = {
    "knife":         "🔪 Knife",
    "scissors":      "✂️ Scissors / Bladed Tool",
    "baseball bat":  "🏏 Blunt Object (Bat)",
    "gun":           "🔫 Firearm",
    "pistol":        "🔫 Handgun",
    "rifle":         "🔫 Rifle",
    "bottle":        "🍾 Potential Blunt Object (Bottle)",
    "cell phone":    "📱 Mobile Device",   # sometimes confused
}

# Colors for bounding boxes (BGR format for OpenCV)
BOX_COLORS = {
    "High":   (0, 0, 255),    # Red   — high confidence
    "Medium": (0, 165, 255),  # Orange — medium confidence
    "Low":    (0, 255, 255),  # Yellow — low confidence
}

# Confidence threshold for detection
CONFIDENCE_THRESHOLD = 0.35

# Path to store / load the model
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "weapon_model.pt")


def load_model():
    """
    Load the YOLOv8 model. Uses yolov8n.pt (nano) by default for speed.
    Downloads the model automatically if not present locally.

    Returns:
        YOLO model object or None if unavailable.
    """
    if not YOLO_AVAILABLE:
        return None

    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        # Use cached path if exists, else auto-download yolov8n
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            print(f"[Weapon] Loaded model from {MODEL_PATH}")
        else:
            model = YOLO("yolov8n.pt")  # Downloads on first run
            # Save to models dir for future use
            model.save(MODEL_PATH)
            print("[Weapon] YOLOv8n model downloaded and saved.")
        return model
    except Exception as e:
        print(f"[Weapon ERROR] Failed to load YOLO model: {e}")
        return None


# --- Load model at module import time (singleton pattern) ---
_model = None

def get_model():
    """Return the singleton YOLO model, loading it if necessary."""
    global _model
    if _model is None:
        _model = load_model()
    return _model


def detect_weapons(image_path: str, output_dir: str = None) -> dict:
    """
    Run weapon detection on a given image file.

    Args:
        image_path (str): Absolute path to the input image.
        output_dir (str): Directory to save the annotated output image.
                          If None, saves in the same directory as input.

    Returns:
        dict: {
            'weapon_detected': 'Yes' | 'No',
            'detections': [{'label': str, 'confidence': float, 'bbox': list}],
            'annotated_image': str (path to annotated image),
            'summary': str
        }
    """
    result = {
        "weapon_detected": "No",
        "detections": [],
        "annotated_image": image_path,
        "summary": "No weapons detected."
    }

    # --- Validate input ---
    if not os.path.exists(image_path):
        result["summary"] = f"ERROR: Image not found at {image_path}"
        return result

    # --- Load image ---
    image = cv2.imread(image_path)
    if image is None:
        result["summary"] = "ERROR: Could not read image file."
        return result

    model = get_model()

    # --- Mock detection if YOLO unavailable ---
    if model is None:
        return _mock_detection(image, image_path, output_dir, result)

    try:
        # Run inference
        results = model(image, conf=CONFIDENCE_THRESHOLD, verbose=False)
        detections = []

        for det in results[0].boxes:
            class_id = int(det.cls[0])
            class_name = model.names[class_id].lower()
            confidence = float(det.conf[0])

            # Only flag weapons or potential weapon classes
            if class_name in WEAPON_CLASSES:
                label = WEAPON_CLASSES[class_name]
                bbox = det.xyxy[0].tolist()  # [x1, y1, x2, y2]

                detections.append({
                    "label": label,
                    "class_name": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": [int(b) for b in bbox]
                })

                # Draw bounding box on image
                x1, y1, x2, y2 = [int(b) for b in bbox]
                conf_tier = _confidence_tier(confidence)
                color = BOX_COLORS[conf_tier]

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                text = f"{label} {confidence:.0%}"
                # Draw label background
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
                cv2.putText(image, text, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if detections:
            result["weapon_detected"] = "Yes"
            labels = [d["label"] for d in detections]
            result["summary"] = f"Detected {len(detections)} weapon(s): {', '.join(labels)}"
        else:
            result["summary"] = "No weapons detected in image."

        result["detections"] = detections

        # Save annotated image
        annotated_path = _save_annotated(image, image_path, output_dir, "weapon")
        result["annotated_image"] = annotated_path

    except Exception as e:
        result["summary"] = f"Detection error: {str(e)}"
        print(f"[Weapon ERROR] {e}")

    return result


def detect_weapons_frame(frame: np.ndarray) -> dict:
    """
    Run weapon detection on an OpenCV frame (numpy array).
    Used for real-time webcam feed analysis.

    Args:
        frame (np.ndarray): BGR image frame from OpenCV.

    Returns:
        dict: Same structure as detect_weapons().
    """
    result = {
        "weapon_detected": "No",
        "detections": [],
        "annotated_frame": frame,
        "summary": "No weapons detected."
    }

    model = get_model()
    if model is None:
        result["summary"] = "YOLO model not available."
        return result

    try:
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        detections = []
        annotated_frame = frame.copy()

        for det in results[0].boxes:
            class_id = int(det.cls[0])
            class_name = model.names[class_id].lower()
            confidence = float(det.conf[0])

            if class_name in WEAPON_CLASSES:
                label = WEAPON_CLASSES[class_name]
                bbox = det.xyxy[0].tolist()
                detections.append({
                    "label": label,
                    "confidence": round(confidence, 3),
                    "bbox": [int(b) for b in bbox]
                })
                x1, y1, x2, y2 = [int(b) for b in bbox]
                color = BOX_COLORS[_confidence_tier(confidence)]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(annotated_frame, f"{label} {confidence:.0%}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if detections:
            result["weapon_detected"] = "Yes"
            result["summary"] = f"Detected {len(detections)} weapon(s)."
        result["detections"] = detections
        result["annotated_frame"] = annotated_frame

    except Exception as e:
        print(f"[Weapon ERROR] Frame detection failed: {e}")

    return result


# ---------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------

def _confidence_tier(confidence: float) -> str:
    """Return confidence tier label for color coding."""
    if confidence >= 0.7:
        return "High"
    elif confidence >= 0.5:
        return "Medium"
    return "Low"


def _save_annotated(image: np.ndarray, original_path: str,
                    output_dir: str, prefix: str) -> str:
    """
    Save an annotated image to disk.

    Args:
        image: Annotated OpenCV image.
        original_path: Original image file path.
        output_dir: Target save directory (defaults to original dir).
        prefix: Filename prefix (e.g., 'weapon').

    Returns:
        str: Path to saved annotated image.
    """
    if output_dir is None:
        output_dir = os.path.dirname(original_path)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_annotated_{timestamp}.jpg"
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, image)
    return save_path


def _mock_detection(image: np.ndarray, image_path: str,
                    output_dir: str, result: dict) -> dict:
    """
    Provide a mock detection result when YOLO is unavailable.
    Used for development/demo purposes.
    """
    print("[Weapon] Using MOCK detection (YOLO not available).")
    result["weapon_detected"] = "Yes (MOCK)"
    result["detections"] = [{
        "label": "🔪 Knife (DEMO)",
        "class_name": "knife",
        "confidence": 0.85,
        "bbox": [50, 50, 250, 200]
    }]
    result["summary"] = "MOCK: Knife detected with 85% confidence (demo mode)"

    # Draw mock box
    cv2.rectangle(image, (50, 50), (250, 200), (0, 0, 255), 3)
    cv2.putText(image, "MOCK: Knife 85%", (55, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    annotated_path = _save_annotated(image, image_path, output_dir, "weapon_mock")
    result["annotated_image"] = annotated_path
    return result
