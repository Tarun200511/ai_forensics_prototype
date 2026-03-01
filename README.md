# 🔍 AI Edge Forensics Prototype
### On-Site Crime Scene Intelligence — Laptop Version

> **Final Year Engineering Project** | AI + Computer Vision + Forensic Analysis

---

## 📌 Project Overview

The **AI Edge Forensics Prototype** is a software-only laptop-based forensic analysis system simulating an AI-powered edge device deployed at crime scenes. It enables investigators to:

- 📷 Capture and analyze crime scene evidence in real-time
- 🔫 Detect weapons using YOLOv8 deep learning
- 🩸 Identify blood stain patterns using HSV color analysis
- 👣 Match footprints using ORB feature descriptors
- 👤 Recognize suspects via face recognition
- 🗂️ Manage cases with a SQLite evidence database

---

## ✨ Features

| Module | Technology | Description |
|--------|-----------|-------------|
| Weapon Detection | YOLOv8 (Ultralytics) | Detects knives, guns, bats in images |
| Blood Pattern Analysis | OpenCV HSV | Classifies blood stains as Impact/Cast-off/Drip |
| Footprint Matching | ORB Feature Matching | Compares shoe prints to stored database |
| Face Recognition | face_recognition + dlib | Identifies suspects from database |
| Case Management | SQLite | Stores all evidence and results |
| Web Dashboard | Flask + Bootstrap 5 | Interactive forensic UI |
| Webcam Capture | OpenCV | Live evidence capture |

---

## 🧱 Tech Stack

- **Language:** Python 3.10+
- **Backend:** Flask 3.x
- **Frontend:** HTML5, CSS3, Bootstrap 5
- **Computer Vision:** OpenCV, YOLOv8 (Ultralytics)
- **Face Recognition:** face_recognition (dlib-based)
- **Database:** SQLite3 (built-in Python)
- **ML:** scikit-learn, NumPy, Pandas

---

## 📂 Project Structure

```
ai_forensics_prototype/
├── app.py                        # Flask application entry point
├── requirements.txt              # Python dependencies
├── README.md
│
├── modules/
│   ├── weapon_detection.py       # YOLOv8 weapon detection
│   ├── blood_pattern_analysis.py # HSV blood stain analysis
│   ├── footprint_analysis.py     # ORB footprint matching
│   ├── face_recognition_module.py# Face detection & recognition
│   └── database.py               # SQLite case management
│
├── models/
│   └── weapon_model.pt           # YOLOv8 model (auto-downloaded)
│
├── static/
│   ├── uploads/                  # Uploaded/captured evidence images
│   ├── css/style.css             # Custom forensic dark theme
│   └── js/webcam.js              # Webcam capture helper
│
├── templates/
│   ├── index.html                # Home page
│   ├── dashboard.html            # Analysis results dashboard
│   └── case_view.html            # Individual case viewer
│
├── suspects/                     # Known suspect face images
│   ├── John_Doe.jpg
│   └── Jane_Smith.jpg
│
└── cases/                        # Auto-created case folders
    ├── case_001/
    └── case_002/
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Pip package manager
- Webcam (optional, for live capture)

### Step 1: Clone or Download
```bash
cd ai_forensics_prototype
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ **Note on dlib/face_recognition:** If `dlib` fails to install, download the prebuilt wheel from [https://github.com/z-mahmud22/Dlib_Windows_Python3.x](https://github.com/z-mahmud22/Dlib_Windows_Python3.x) and install manually.

### Step 4: Add Suspect Images (Optional)
Place suspect face images in the `/suspects/` folder:
```
suspects/
├── John_Doe.jpg
├── Jane_Smith.jpg
```

### Step 5: Run the Application
```bash
python app.py
```

Open your browser and navigate to: **http://127.0.0.1:5000**

---

## 🖥️ Usage Guide

1. **Home Page** → Choose to upload an image or capture from webcam
2. **Upload/Capture** → Select one or more analysis modules to run
3. **Run Analysis** → System runs selected AI modules
4. **Dashboard** → View annotated results with bounding boxes
5. **Case Stored** → Evidence saved automatically to SQLite

---

## 📸 Screenshots

> *(Add screenshots of the dashboard and results here)*

| Home Page | Analysis Dashboard | Case View |
|-----------|-------------------|-----------|
| ![Home](static/uploads/screenshot_home.png) | ![Dashboard](static/uploads/screenshot_dashboard.png) | ![Case](static/uploads/screenshot_case.png) |

---

## 🔬 Module Details

### 🔫 Weapon Detection
Uses **YOLOv8** pretrained on COCO dataset. Detects:
- Knives (`knife` class)
- Firearms (`pistol`, `rifle` via fine-tuned or COCO)
- Sports equipment as blunt weapons

### 🩸 Blood Pattern Analysis
HSV color range detection in `[0°–10°]` and `[160°–180°]` hue channels.
Pattern classification rules:
- **Area < 500 px** → Impact spatter
- **Aspect ratio > 3:1** → Cast-off pattern  
- **Circularity > 0.7** → Drip pattern

### 👣 Footprint Matching
- Converts image to grayscale + Canny edge detection
- Extracts **ORB keypoint descriptors**
- BFMatcher + ratio test for comparison
- Returns match percentage

### 👤 Face Recognition
- Detects faces using HOG + SVM (dlib)
- Creates 128-dimensional face encodings
- Compares against suspect database encodings
- Returns name + confidence %

---

## 🔮 Future Scope

- **Raspberry Pi Deployment** — Edge deployment for field use
- **Firebase Sync** — Cloud case sync for multi-unit teams
- **GPS Tagging** — Geolocation metadata on evidence photos
- **Chain of Custody Logging** — Tamper-evident evidence trail
- **3D Scene Reconstruction** — Point cloud from stereo images
- **GSR Detection** — Gunshot residue pattern recognition
- **Thermal Imaging** — Infrared camera integration

---

## 👨‍💻 Developer

**Final Year Engineering Project**  
AI Edge Forensics Prototype — On-Site Crime Scene Intelligence

---

## 📄 License

For academic/educational use only. Not intended for production forensic deployment.
