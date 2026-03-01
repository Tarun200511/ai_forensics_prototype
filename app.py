"""
app.py — AI Edge Forensics Prototype — Flask Application Entry Point
=======================================================================
Main Flask web application that orchestrates:
  - Evidence capture (webcam / image upload)
  - AI module execution (weapon, blood, footprint, face)
  - Case creation and database storage
  - Web dashboard rendering

Usage:
    python app.py
Then navigate to: http://127.0.0.1:5000
"""

import os
import cv2
import json
import uuid
import shutil
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect,
    url_for, jsonify, flash, send_from_directory
)
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------
# Module Imports
# ---------------------------------------------------------------
from modules.database import (
    init_db, create_case, update_case_results,
    get_all_cases, get_case_by_id, get_case_log,
    log_analysis_event, delete_case
)
from modules.weapon_detection import detect_weapons
from modules.blood_pattern_analysis import analyze_blood_pattern
from modules.footprint_analysis import match_footprint, build_footprint_database
from modules.face_recognition_module import recognize_faces, load_suspects_database


# ---------------------------------------------------------------
# Flask App Initialization
# ---------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "forensics_prototype_secret_2024"

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER   = os.path.join(BASE_DIR, "static", "uploads")
CASES_FOLDER    = os.path.join(BASE_DIR, "cases")
FOOTPRINTS_DIR  = os.path.join(BASE_DIR, "static", "footprints")
SUSPECTS_DIR    = os.path.join(BASE_DIR, "suspects")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MB

app.config["UPLOAD_FOLDER"]     = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ---------------------------------------------------------------
# Ensure directories exist
# ---------------------------------------------------------------
for d in [UPLOAD_FOLDER, CASES_FOLDER, FOOTPRINTS_DIR, SUSPECTS_DIR,
          os.path.join(BASE_DIR, "models")]:
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------
# Startup Initialization
# ---------------------------------------------------------------
def initialize_app():
    """Run startup tasks: database init, suspects loading."""
    print("[App] Initializing AI Edge Forensics Prototype...")
    init_db()
    load_suspects_database(SUSPECTS_DIR)
    build_footprint_database(FOOTPRINTS_DIR)
    print("[App] Ready. Navigate to http://127.0.0.1:5000")


# ---------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    """Check if uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_case_id() -> str:
    """Generate a unique case ID with date prefix."""
    date_str = datetime.now().strftime("%Y%m%d")
    short_uid = str(uuid.uuid4())[:8].upper()
    return f"CASE_{date_str}_{short_uid}"


def get_relative_path(absolute_path: str) -> str:
    """
    Convert an absolute path to a web-accessible relative path
    (relative to /static/).

    Args:
        absolute_path: Full filesystem path.

    Returns:
        str: Web path like 'uploads/filename.jpg'
    """
    try:
        static_dir = os.path.join(BASE_DIR, "static")
        rel = os.path.relpath(absolute_path, static_dir)
        return rel.replace("\\", "/")
    except Exception:
        return absolute_path


# ---------------------------------------------------------------
# Routes — Home
# ---------------------------------------------------------------

@app.route("/")
def index():
    """
    Home page — displays navigation buttons and recent case count.
    """
    all_cases = get_all_cases()
    recent_cases = all_cases[:5]  # Show 5 most recent
    stats = {
        "total_cases": len(all_cases),
        "weapons_found": sum(1 for c in all_cases if c.get("weapon_detected") == "Yes"),
        "blood_found": sum(1 for c in all_cases if c.get("blood_detected") == "Yes"),
        "faces_matched": sum(1 for c in all_cases if c.get("face_match") == "Yes"),
    }
    return render_template("index.html", stats=stats, recent_cases=recent_cases)


# ---------------------------------------------------------------
# Routes — Webcam Capture
# ---------------------------------------------------------------

@app.route("/capture", methods=["GET", "POST"])
def capture():
    """
    Webcam capture page. On POST, captures a frame from the webcam
    and saves it to /static/uploads/.
    """
    if request.method == "POST":
        try:
            cap = cv2.VideoCapture(0)  # Camera index 0 = default webcam
            if not cap.isOpened():
                flash("❌ Webcam not accessible. Please try uploading an image instead.", "error")
                return redirect(url_for("index"))

            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                flash("❌ Failed to capture frame from webcam.", "error")
                return redirect(url_for("index"))

            # Save captured frame
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{ts}.jpg"
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            cv2.imwrite(save_path, frame)

            flash(f"✅ Frame captured: {filename}", "success")
            return redirect(url_for("analyze_page", filename=filename))

        except Exception as e:
            flash(f"❌ Webcam error: {str(e)}", "error")
            return redirect(url_for("index"))

    return render_template("capture.html")


@app.route("/webcam-feed")
def webcam_feed():
    """
    Returns a single webcam frame as JPEG for live preview.
    Called by JavaScript in the capture page.
    """
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            _, buffer = cv2.imencode(".jpg", frame)
            return buffer.tobytes(), 200, {
                "Content-Type": "image/jpeg",
                "Cache-Control": "no-cache"
            }
    except Exception as e:
        print(f"[Webcam] Error: {e}")

    return "", 204


# ---------------------------------------------------------------
# Routes — File Upload
# ---------------------------------------------------------------

@app.route("/upload", methods=["GET", "POST"])
def upload():
    """
    Image upload page. Accepts JPG/PNG evidence images.
    Redirects to the analysis setup page on success.
    """
    if request.method == "POST":
        if "file" not in request.files:
            flash("❌ No file selected.", "error")
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            flash("❌ Empty filename.", "error")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash(f"❌ File type not allowed. Use: {', '.join(ALLOWED_EXTENSIONS)}", "error")
            return redirect(request.url)

        # Secure the filename and save
        filename = secure_filename(file.filename)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = ts + filename
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        flash(f"✅ Image uploaded: {filename}", "success")
        return redirect(url_for("analyze_page", filename=filename))

    return render_template("upload.html")


# ---------------------------------------------------------------
# Routes — Analysis Setup
# ---------------------------------------------------------------

@app.route("/analyze")
def analyze_page():
    """
    Analysis setup page — shows the uploaded image and lets user
    select which forensic modules to run.
    """
    filename = request.args.get("filename", "")
    image_path = os.path.join(UPLOAD_FOLDER, filename)

    if not filename or not os.path.exists(image_path):
        flash("❌ Image not found. Please upload or capture an image first.", "error")
        return redirect(url_for("index"))

    return render_template("analyze.html", filename=filename, image_url=url_for("static", filename=f"uploads/{filename}"))


# ---------------------------------------------------------------
# Routes — Run Analysis (API endpoint)
# ---------------------------------------------------------------

@app.route("/run-analysis", methods=["POST"])
def run_analysis():
    """
    Core analysis endpoint. Receives selected modules and image filename,
    runs AI modules, stores results in SQLite, and returns JSON.

    POST Body (form data):
        filename: str
        modules: list of ['weapon', 'blood', 'footprint', 'face']

    Returns:
        JSON with analysis results and case_id.
    """
    filename = request.form.get("filename", "")
    selected_modules = request.form.getlist("modules")

    image_path = os.path.join(UPLOAD_FOLDER, filename)

    if not filename or not os.path.exists(image_path):
        return jsonify({"error": "Image not found."}), 400

    if not selected_modules:
        return jsonify({"error": "No analysis modules selected."}), 400

    # --- Generate Case ID and create case folder ---
    case_id = generate_case_id()
    case_folder = os.path.join(CASES_FOLDER, case_id)
    os.makedirs(case_folder, exist_ok=True)

    # Copy original image to case folder
    case_image_path = os.path.join(case_folder, filename)
    shutil.copy2(image_path, case_image_path)

    # Create database record
    create_case(case_id, image_path=image_path)

    results = {
        "case_id": case_id,
        "original_image": url_for("static", filename=f"uploads/{filename}"),
        "modules_run": selected_modules
    }

    db_update = {}

    # -------------------------------------------------------
    # Module 1: Weapon Detection
    # -------------------------------------------------------
    if "weapon" in selected_modules:
        try:
            weapon_result = detect_weapons(image_path, output_dir=UPLOAD_FOLDER)
            results["weapon"] = {
                "detected": weapon_result["weapon_detected"],
                "summary": weapon_result["summary"],
                "detections": weapon_result["detections"],
                "annotated_image": url_for("static", filename=get_relative_path(
                    weapon_result["annotated_image"]))
                    if os.path.exists(weapon_result["annotated_image"]) else results["original_image"]
            }
            db_update["weapon_detected"] = weapon_result["weapon_detected"]
            db_update["weapon_labels"] = ", ".join(
                [d["label"] for d in weapon_result["detections"]]
            ) or "None"
            log_analysis_event(case_id, "weapon_detection", weapon_result["summary"])
        except Exception as e:
            results["weapon"] = {"error": str(e), "detected": "Error"}
            print(f"[App] Weapon detection error: {e}")

    # -------------------------------------------------------
    # Module 2: Blood Pattern Analysis
    # -------------------------------------------------------
    if "blood" in selected_modules:
        try:
            blood_result = analyze_blood_pattern(image_path, output_dir=UPLOAD_FOLDER)
            results["blood"] = {
                "detected": blood_result["blood_detected"],
                "pattern": blood_result["pattern_type"],
                "region_count": blood_result["region_count"],
                "summary": blood_result["summary"],
                "annotated_image": url_for("static", filename=get_relative_path(
                    blood_result["annotated_image"]))
                    if os.path.exists(blood_result["annotated_image"]) else results["original_image"]
            }
            db_update["blood_detected"] = blood_result["blood_detected"]
            db_update["blood_pattern"] = blood_result["pattern_type"]
            log_analysis_event(case_id, "blood_analysis", blood_result["summary"])
        except Exception as e:
            results["blood"] = {"error": str(e), "detected": "Error"}
            print(f"[App] Blood analysis error: {e}")

    # -------------------------------------------------------
    # Module 3: Footprint Matching
    # -------------------------------------------------------
    if "footprint" in selected_modules:
        try:
            fp_result = match_footprint(image_path, output_dir=UPLOAD_FOLDER)
            results["footprint"] = {
                "match": fp_result["footprint_match"],
                "match_id": fp_result["match_id"],
                "score": fp_result["match_score"],
                "summary": fp_result["summary"],
                "all_scores": fp_result["all_scores"],
                "annotated_image": url_for("static", filename=get_relative_path(
                    fp_result["annotated_image"]))
                    if os.path.exists(fp_result["annotated_image"]) else results["original_image"]
            }
            db_update["footprint_match"] = fp_result["footprint_match"]
            db_update["footprint_score"] = fp_result["match_score"]
            log_analysis_event(case_id, "footprint_matching", fp_result["summary"])
        except Exception as e:
            results["footprint"] = {"error": str(e), "match": "Error"}
            print(f"[App] Footprint error: {e}")

    # -------------------------------------------------------
    # Module 4: Face Recognition
    # -------------------------------------------------------
    if "face" in selected_modules:
        try:
            face_result = recognize_faces(image_path, output_dir=UPLOAD_FOLDER)
            results["face"] = {
                "detected": face_result["face_detected"],
                "count": face_result["face_count"],
                "match": face_result["face_match"],
                "names": face_result["face_names"],
                "matches": face_result["matches"],
                "summary": face_result["summary"],
                "annotated_image": url_for("static", filename=get_relative_path(
                    face_result["annotated_image"]))
                    if os.path.exists(face_result["annotated_image"]) else results["original_image"]
            }
            db_update["face_match"] = face_result["face_match"]
            db_update["face_name"] = face_result["face_names"]
            log_analysis_event(case_id, "face_recognition", face_result["summary"])
        except Exception as e:
            results["face"] = {"error": str(e), "detected": "Error"}
            print(f"[App] Face recognition error: {e}")

    # --- Update case in database ---
    update_case_results(case_id, db_update)

    # --- Save full results JSON to case folder ---
    results_json_path = os.path.join(case_folder, "results.json")
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return jsonify(results)


# ---------------------------------------------------------------
# Routes — Dashboard (shows results after analysis)
# ---------------------------------------------------------------

@app.route("/dashboard/<case_id>")
def dashboard(case_id: str):
    """
    Render the analysis results dashboard for a specific case.

    Args:
        case_id: Case identifier from URL.
    """
    case = get_case_by_id(case_id)
    if not case:
        flash(f"❌ Case '{case_id}' not found.", "error")
        return redirect(url_for("cases"))

    case_log = get_case_log(case_id)

    # Load full results JSON if available
    results_json_path = os.path.join(CASES_FOLDER, case_id, "results.json")
    full_results = {}
    if os.path.exists(results_json_path):
        try:
            with open(results_json_path, "r") as f:
                full_results = json.load(f)
        except Exception:
            pass

    return render_template("dashboard.html",
                           case=case,
                           case_log=case_log,
                           full_results=full_results)


# ---------------------------------------------------------------
# Routes — Case List
# ---------------------------------------------------------------

@app.route("/cases")
def cases():
    """
    Display all forensic cases stored in the database.
    """
    all_cases = get_all_cases()
    return render_template("case_view.html", cases=all_cases)


# ---------------------------------------------------------------
# Routes — Case Detail
# ---------------------------------------------------------------

@app.route("/case/<case_id>")
def case_detail(case_id: str):
    """
    Show detailed view for a single case (redirect to dashboard).
    """
    return redirect(url_for("dashboard", case_id=case_id))


# ---------------------------------------------------------------
# Routes — Delete Case
# ---------------------------------------------------------------

@app.route("/case/<case_id>/delete", methods=["POST"])
def delete_case_route(case_id: str):
    """
    Delete a case and its associated files.
    """
    success = delete_case(case_id)
    case_folder = os.path.join(CASES_FOLDER, case_id)
    if os.path.exists(case_folder):
        shutil.rmtree(case_folder, ignore_errors=True)

    if success:
        flash(f"✅ Case {case_id} deleted.", "success")
    else:
        flash(f"❌ Failed to delete case {case_id}.", "error")

    return redirect(url_for("cases"))


# ---------------------------------------------------------------
# Routes — API Endpoints
# ---------------------------------------------------------------

@app.route("/api/cases")
def api_cases():
    """JSON API endpoint returning all cases."""
    return jsonify(get_all_cases())


@app.route("/api/case/<case_id>")
def api_case(case_id: str):
    """JSON API endpoint returning a single case."""
    return jsonify(get_case_by_id(case_id))


@app.route("/api/stats")
def api_stats():
    """JSON API endpoint with aggregate statistics."""
    all_cases = get_all_cases()
    return jsonify({
        "total_cases": len(all_cases),
        "weapons_found": sum(1 for c in all_cases if c.get("weapon_detected") == "Yes"),
        "blood_found": sum(1 for c in all_cases if c.get("blood_detected") == "Yes"),
        "footprints_matched": sum(1 for c in all_cases if c.get("footprint_match") == "Yes"),
        "faces_matched": sum(1 for c in all_cases if c.get("face_match") == "Yes"),
    })


# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------

if __name__ == "__main__":
    initialize_app()
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False  # Prevent double initialization
    )
