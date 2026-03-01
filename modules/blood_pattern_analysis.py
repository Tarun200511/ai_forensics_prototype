"""
blood_pattern_analysis.py — Forensic Blood Stain Pattern Analysis
AI Edge Forensics Prototype

Detects red blood-coloured regions using HSV color space analysis
and classifies the stain pattern type using contour morphology.
"""

import cv2
import numpy as np
import os
from datetime import datetime


# ---------------------------------------------------------------
# HSV Color Range for Blood / Dark Red Detection
# Hue range: 0–10° and 160–180° (red wraps around in HSV)
# ---------------------------------------------------------------
BLOOD_LOWER_1 = np.array([0, 50, 30], dtype=np.uint8)
BLOOD_UPPER_1 = np.array([12, 255, 200], dtype=np.uint8)

BLOOD_LOWER_2 = np.array([160, 50, 30], dtype=np.uint8)
BLOOD_UPPER_2 = np.array([180, 255, 200], dtype=np.uint8)

# Minimum contour area to consider as a stain (pixels²)
MIN_STAIN_AREA = 150

# Pattern classification thresholds
IMPACT_MAX_AREA = 600          # Small spots → Impact spatter
CAST_OFF_ASPECT_RATIO = 2.8    # Long thin shape → Cast-off
DRIP_CIRCULARITY = 0.65        # Round shape → Drip pattern


def analyze_blood_pattern(image_path: str, output_dir: str = None) -> dict:
    """
    Analyze an image for blood stain patterns.

    Steps:
        1. Convert image to HSV color space
        2. Mask red-range pixels (blood color)
        3. Apply morphological cleanup
        4. Find contours
        5. Classify pattern type per contour
        6. Draw annotated results

    Args:
        image_path (str): Path to the input evidence image.
        output_dir (str): Directory to save annotated output. Defaults to input dir.

    Returns:
        dict: {
            'blood_detected': 'Yes' | 'No',
            'pattern_type': str,
            'region_count': int,
            'annotated_image': str (path),
            'details': list of per-contour classifications,
            'summary': str
        }
    """
    result = {
        "blood_detected": "No",
        "pattern_type": "None",
        "region_count": 0,
        "annotated_image": image_path,
        "details": [],
        "summary": "No blood stain regions detected."
    }

    # --- Validate input image ---
    if not os.path.exists(image_path):
        result["summary"] = f"ERROR: File not found: {image_path}"
        return result

    image = cv2.imread(image_path)
    if image is None:
        result["summary"] = "ERROR: Could not load image."
        return result

    # --- Step 1: Resize for consistent processing ---
    h, w = image.shape[:2]
    scale = min(1200 / w, 1200 / h, 1.0)
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    annotated = image.copy()

    # --- Step 2: Convert to HSV ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --- Step 3: Create blood color mask (two red hue ranges) ---
    mask1 = cv2.inRange(hsv, BLOOD_LOWER_1, BLOOD_UPPER_1)
    mask2 = cv2.inRange(hsv, BLOOD_LOWER_2, BLOOD_UPPER_2)
    blood_mask = cv2.bitwise_or(mask1, mask2)

    # --- Step 4: Morphological operations (reduce noise) ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blood_mask = cv2.morphologyEx(blood_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    blood_mask = cv2.morphologyEx(blood_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    blood_mask = cv2.dilate(blood_mask, kernel, iterations=1)

    # --- Step 5: Find contours ---
    contours, _ = cv2.findContours(blood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by minimum area
    valid_contours = [c for c in contours if cv2.contourArea(c) >= MIN_STAIN_AREA]

    if not valid_contours:
        # No valid stains found
        annotated_path = _save_annotated(annotated, image_path, output_dir, "blood")
        result["annotated_image"] = annotated_path
        return result

    # --- Step 6: Classify each stain region ---
    pattern_votes = {"Impact": 0, "Cast-off": 0, "Drip": 0, "Unknown": 0}
    details = []

    for i, contour in enumerate(valid_contours):
        area = cv2.contourArea(contour)
        pattern = _classify_contour(contour, area)
        pattern_votes[pattern] += 1

        # Draw contour and label
        _draw_stain(annotated, contour, pattern, i + 1)

        # Bounding rect for reporting
        x, y, cw, ch = cv2.boundingRect(contour)
        details.append({
            "region_id": i + 1,
            "area_px": int(area),
            "pattern": pattern,
            "bbox": [x, y, x + cw, y + ch]
        })

    # --- Step 7: Determine overall dominant pattern ---
    dominant = _dominant_pattern(pattern_votes, len(valid_contours))

    # Save annotated image
    annotated_path = _save_annotated(annotated, image_path, output_dir, "blood")

    result["blood_detected"] = "Yes"
    result["pattern_type"] = dominant
    result["region_count"] = len(valid_contours)
    result["annotated_image"] = annotated_path
    result["details"] = details
    result["summary"] = (
        f"Blood detected. Pattern: {dominant}. "
        f"Regions found: {len(valid_contours)}."
    )

    return result


# ---------------------------------------------------------------
# Pattern Classification Logic
# ---------------------------------------------------------------

def _classify_contour(contour: np.ndarray, area: float) -> str:
    """
    Classify a single blood stain contour into a forensic pattern.

    Rules:
        - Small area (<= IMPACT_MAX_AREA) → Impact spatter
        - High aspect ratio (>= CAST_OFF_ASPECT_RATIO) → Cast-off
        - High circularity (>= DRIP_CIRCULARITY) → Drip
        - Otherwise → Unknown

    Args:
        contour: OpenCV contour array.
        area: Contour area in pixels².

    Returns:
        str: Pattern name.
    """
    # Impact — small droplets
    if area <= IMPACT_MAX_AREA:
        return "Impact"

    # Fit bounding rect to compute aspect ratio
    _, _, w, h = cv2.boundingRect(contour)
    if h == 0:
        return "Unknown"

    aspect_ratio = max(w, h) / min(w, h)

    # Cast-off — elongated / linear shape
    if aspect_ratio >= CAST_OFF_ASPECT_RATIO:
        return "Cast-off"

    # Circularity = 4π × area / perimeter²
    perimeter = cv2.arcLength(contour, True)
    if perimeter > 0:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity >= DRIP_CIRCULARITY:
            return "Drip"

    return "Unknown"


def _dominant_pattern(votes: dict, total: int) -> str:
    """
    Determine the dominant stain pattern from classification votes.

    Args:
        votes (dict): Pattern → count mapping.
        total (int): Total number of stain regions.

    Returns:
        str: Dominant pattern label with description.
    """
    if total == 0:
        return "None"

    # Get pattern with most votes (excluding Unknown)
    primary_votes = {k: v for k, v in votes.items() if k != "Unknown"}
    if not any(primary_votes.values()):
        return "Unknown Pattern"

    dominant = max(primary_votes, key=primary_votes.get)

    descriptions = {
        "Impact":   "Impact Spatter — High-velocity blood droplets from force",
        "Cast-off": "Cast-off Pattern — Blood slung from moving weapon",
        "Drip":     "Drip Pattern — Passive blood drip from gravity",
    }
    return descriptions.get(dominant, dominant)


def _draw_stain(image: np.ndarray, contour: np.ndarray,
                pattern: str, region_id: int):
    """
    Draw contour outline and pattern label on the annotated image.

    Args:
        image: BGR image to draw on (modified in-place).
        contour: Contour to draw.
        pattern: Pattern type string.
        region_id: Region number for labeling.
    """
    color_map = {
        "Impact":   (0, 100, 255),    # Orange
        "Cast-off": (0, 0, 200),      # Dark red
        "Drip":     (200, 50, 200),   # Purple
        "Unknown":  (128, 128, 128),  # Gray
    }
    color = color_map.get(pattern, (255, 255, 255))

    # Draw filled contour (semi-transparent via addWeighted)
    overlay = image.copy()
    cv2.drawContours(overlay, [contour], -1, color, -1)
    cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

    # Draw contour outline
    cv2.drawContours(image, [contour], -1, color, 2)

    # Label at centroid
    M = cv2.moments(contour)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        label = f"#{region_id} {pattern}"
        cv2.putText(image, label, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def _save_annotated(image: np.ndarray, original_path: str,
                    output_dir: str, prefix: str) -> str:
    """Save annotated image and return its path."""
    if output_dir is None:
        output_dir = os.path.dirname(original_path)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_annotated_{timestamp}.jpg"
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, image)
    return save_path


def get_blood_mask_preview(image_path: str) -> np.ndarray:
    """
    Return a binary mask image showing blood-colored regions.
    Useful for debugging and UI preview.

    Args:
        image_path (str): Path to input image.

    Returns:
        np.ndarray: Binary mask or None on error.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, BLOOD_LOWER_1, BLOOD_UPPER_1)
    mask2 = cv2.inRange(hsv, BLOOD_LOWER_2, BLOOD_UPPER_2)
    combined = cv2.bitwise_or(mask1, mask2)

    # Return as BGR for consistent display
    return cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
