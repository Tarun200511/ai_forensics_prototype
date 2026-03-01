"""
test_all_modules.py — Comprehensive feature test runner
AI Edge Forensics Prototype

Runs all 5 modules against sample images and prints a detailed report.
Usage: python test_all_modules.py
"""
import os, sys, traceback

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results = {}

print("=" * 60)
print("AI Edge Forensics — Full Module Test Suite")
print("=" * 60)

# ── Test 1: Database ─────────────────────────────────────────
print("\n[1] DATABASE MODULE")
try:
    from modules.database import init_db, create_case, update_case_results, get_all_cases, get_case_by_id, delete_case
    init_db()
    cid = "TEST_CASE_001"
    assert create_case(cid, "test.jpg"), "create_case failed"
    assert update_case_results(cid, {"weapon_detected": "Yes", "blood_detected": "Yes"}), "update failed"
    case = get_case_by_id(cid)
    assert case["weapon_detected"] == "Yes", "read-back mismatch"
    all_c = get_all_cases()
    assert any(c["case_id"] == cid for c in all_c), "case not in list"
    delete_case(cid)
    print(f"  {PASS} init_db, create_case, update, get_by_id, get_all, delete — all OK")
    results["database"] = "PASS"
except Exception as e:
    print(f"  {FAIL} {e}")
    traceback.print_exc()
    results["database"] = f"FAIL: {e}"

# ── Test 2: Blood Pattern Analysis ───────────────────────────
print("\n[2] BLOOD PATTERN ANALYSIS")
try:
    from modules.blood_pattern_analysis import analyze_blood_pattern
    img_path = os.path.join(BASE, "static", "uploads", "test_blood_stain.jpg")
    if not os.path.exists(img_path):
        print(f"  {WARN} test image missing, regenerating...")
        os.system(f'python "{os.path.join(BASE, "generate_test_data.py")}"')
    assert os.path.exists(img_path), f"Test image still missing: {img_path}"
    out_dir = os.path.join(BASE, "static", "uploads")
    r = analyze_blood_pattern(img_path, output_dir=out_dir)
    print(f"  blood_detected: {r['blood_detected']}")
    print(f"  pattern_type:   {r['pattern_type']}")
    print(f"  region_count:   {r['region_count']}")
    print(f"  annotated_img:  {os.path.basename(r['annotated_image'])}")
    assert os.path.exists(r["annotated_image"]), "Annotated image not saved"
    print(f"  {PASS} Blood analysis OK — detected={r['blood_detected']}, regions={r['region_count']}")
    results["blood"] = "PASS"
except Exception as e:
    print(f"  {FAIL} {e}")
    traceback.print_exc()
    results["blood"] = f"FAIL: {e}"

# ── Test 3: Footprint Analysis ───────────────────────────────
print("\n[3] FOOTPRINT MATCHING")
try:
    from modules.footprint_analysis import build_footprint_database, match_footprint
    fp_dir = os.path.join(BASE, "static", "footprints")
    if not os.listdir(fp_dir):
        print(f"  {WARN} No reference prints, regenerating...")
        os.system(f'python "{os.path.join(BASE, "generate_test_data.py")}"')
    db = build_footprint_database(fp_dir)
    print(f"  DB entries: {len(db)}")
    query = os.path.join(BASE, "static", "uploads", "test_footprint.jpg")
    out_dir = os.path.join(BASE, "static", "uploads")
    r = match_footprint(query, output_dir=out_dir)
    print(f"  footprint_match: {r['footprint_match']}")
    print(f"  match_id:        {r['match_id']}")
    print(f"  match_score:     {r['match_score']}%")
    print(f"  all_scores:      {r['all_scores']}")
    assert r["match_score"] >= 0, "negative score"
    print(f"  {PASS} Footprint matching OK")
    results["footprint"] = "PASS"
except Exception as e:
    print(f"  {FAIL} {e}")
    traceback.print_exc()
    results["footprint"] = f"FAIL: {e}"

# ── Test 4: Weapon Detection ─────────────────────────────────
print("\n[4] WEAPON DETECTION (YOLOv8)")
try:
    from modules.weapon_detection import detect_weapons, get_model
    img_path = os.path.join(BASE, "static", "uploads", "test_scene.jpg")
    out_dir = os.path.join(BASE, "static", "uploads")
    model = get_model()
    if model:
        print(f"  Model loaded: {type(model).__name__}")
    else:
        print(f"  {WARN} YOLO model not available, mock mode active")
    r = detect_weapons(img_path, output_dir=out_dir)
    print(f"  weapon_detected: {r['weapon_detected']}")
    print(f"  detections: {len(r['detections'])}")
    print(f"  summary: {r['summary']}")
    print(f"  {PASS} Weapon detection OK (mock or real)")
    results["weapon"] = "PASS"
except Exception as e:
    print(f"  {FAIL} {e}")
    traceback.print_exc()
    results["weapon"] = f"FAIL: {e}"

# ── Test 5: Face Recognition ─────────────────────────────────
print("\n[5] FACE RECOGNITION")
try:
    from modules.face_recognition_module import load_suspects_database, recognize_faces
    suspects_dir = os.path.join(BASE, "suspects")
    loaded = load_suspects_database(suspects_dir)
    print(f"  Suspects loaded: {loaded}")
    img_path = os.path.join(BASE, "static", "uploads", "test_scene.jpg")
    out_dir = os.path.join(BASE, "static", "uploads")
    r = recognize_faces(img_path, output_dir=out_dir)
    print(f"  face_detected: {r['face_detected']}")
    print(f"  face_count: {r['face_count']}")
    print(f"  face_match: {r['face_match']}")
    print(f"  summary: {r['summary']}")
    print(f"  {PASS} Face recognition OK (mock or real)")
    results["face"] = "PASS"
except Exception as e:
    print(f"  {FAIL} {e}")
    traceback.print_exc()
    results["face"] = f"FAIL: {e}"

# ── Test 6: Flask Routes availability ───────────────────────
print("\n[6] FLASK APP IMPORT")
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", os.path.join(BASE, "app.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    flask_app = mod.app
    client = flask_app.test_client()
    flask_app.config["TESTING"] = True

    # Test home route
    resp = client.get("/")
    assert resp.status_code == 200, f"Home page returned {resp.status_code}"
    print(f"  {PASS} GET /  → 200 OK")

    # Test cases route
    resp = client.get("/cases")
    assert resp.status_code == 200, f"/cases returned {resp.status_code}"
    print(f"  {PASS} GET /cases  → 200 OK")

    # Test upload page
    resp = client.get("/upload")
    assert resp.status_code == 200
    print(f"  {PASS} GET /upload → 200 OK")

    # Test API stats
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    import json
    data = json.loads(resp.data)
    print(f"  {PASS} GET /api/stats → {data}")

    # Test full analysis pipeline via test client (all 4 modules)
    test_img = os.path.join(BASE, "static", "uploads", "test_blood_stain.jpg")
    with open(test_img, "rb") as f:
        from io import BytesIO
        img_bytes = f.read()
    
    # First upload the file
    resp = client.post("/upload", data={
        "file": (BytesIO(img_bytes), "test_blood_stain.jpg")
    }, content_type="multipart/form-data", follow_redirects=False)
    print(f"  POST /upload → {resp.status_code}")

    # Now run analysis
    resp = client.post("/run-analysis", data={
        "filename": "test_blood_stain.jpg",
        "modules": ["weapon", "blood", "footprint", "face"]
    })
    assert resp.status_code == 200, f"/run-analysis returned {resp.status_code}: {resp.data[:200]}"
    analysis = json.loads(resp.data)
    print(f"  {PASS} POST /run-analysis → case_id={analysis.get('case_id')}")

    # Test dashboard
    cid = analysis.get("case_id")
    resp = client.get(f"/dashboard/{cid}")
    assert resp.status_code == 200, f"Dashboard returned {resp.status_code}"
    print(f"  {PASS} GET /dashboard/{cid} → 200 OK")

    results["flask"] = "PASS"
except Exception as e:
    print(f"  {FAIL} {e}")
    traceback.print_exc()
    results["flask"] = f"FAIL: {e}"

# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
for mod, res in results.items():
    icon = "✅" if "PASS" in res else "❌"
    print(f"  {icon}  {mod:<15} {res}")

fails = [k for k, v in results.items() if "FAIL" in v]
if fails:
    print(f"\n❌ FAILING: {', '.join(fails)}")
    sys.exit(1)
else:
    print("\n✅ ALL TESTS PASSED")
