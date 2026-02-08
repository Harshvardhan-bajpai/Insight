# 🎥 INSIGHT: Intelligent Surveillance and Guidance Technology

An advanced real-time surveillance system that leverages computer vision and AI to detect threats, identify persons, and guide automated response systems. INSIGHT integrates multiple video feeds (CCTV, Drone, Rover) with intelligent analytics for enhanced security and crowd management.

## 🌟 Features

### 📊 Multi-Feed Analytics
- **CCTV Feed**: Stationary camera feed for area monitoring
- **Drone Feed**: Aerial perspective for crowd and area surveillance
- **Rover Feed**: Mobile ground-level detection with autonomous tracking

### 🔍 Advanced Detection Modules
- **Person Detection & Tracking**: Real-time detection and ID tracking of individuals across frames
- **Face Detection & Recognition**: Identifies and recognizes faces against a database
- **Altercation Detection**: Detects aggressive behavior, pushing, rapid movements
- **Trespass Detection**: Alerts when individuals enter restricted zones
- **Crowd Analysis**: Monitors crowd density, panic detection, and stampede prediction
- **Attribute Recognition**: Extracts person attributes (clothing, pose, etc.)
- **Hand Gesture Detection**: Identifies waving and other hand signals

### 🚀 Autonomous Systems
- **Rover Tracking System**: Auto-follows detected persons with command-based control
- **Alert Engine**: Intelligent event queuing with cooldown-based alert management
- **Event Database**: Persistent storage of all detected events with JSON format

### 🌐 Web Dashboard
- Real-time stream visualization for all three feeds
- Live alert notifications and event history
- Control panels for CCTV, Drone, and Rover
- Face database management interface
- System statistics and monitoring

## 📋 Project Structure

```
INSIGHT/
├── app.py                       # Flask web server & dashboard
├── main_core.py                 # Core system orchestrator
├── person_detection.py          # YOLOv8-based person detection & ├── face_detection.py            # YuNet face detection model
├── face_recognition_core.py     # InsightFace embeddings & matching
├── face_enrollment.py           # Face database enrollment
├── altercation_detection.py     # Aggressive behavior detection
├── trespass_detection.py        # Restricted zone violation 
├── attributes.py                # Person attribute extraction
├── crowd_analysis.py            # Crowd density & panic detection
├── waving_detection.py          # MediaPipe-based gesture detection
├── rover_face_watch.py          # Rover-specific face matching
├── rover_tracking_system.py     # Autonomous rover control logic
├── alert_engine.py              # Event processing & alerting
│
├── templates/
│   ├── controlpanel.html        # Main CCTV control dashboard
│   ├── roverpanel.html          # Rover control interface
│   ├── dronepanel.html          # Drone feed viewer
│   ├── facedatabase.html        # Face enrollment & management
│   ├── events.html              # Event history & alerts
│   └── cctvpanel.html           # CCTV-specific controls
└── static/                      # CSS, JS, images
├── events_db.json               # Event database
├── faces/
│   ├── face_db.json             # Face embeddings & metadata
│   └── embeddings.npy           # Face embedding vectors
│
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- OpenCV 4.8+
- CUDA-capable GPU (optional, for faster processing)
- Windows/Linux/MacOS

### Step 1: Clone the Repository
```bash
git clone https://github.com/Harshvardhan-bajpai/Insight.git
cd Insight
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configuration
Edit `app.py` to set up your hardware connections:
```python
# Line 39: Change to your rover serial port
ROVER_SERIAL_PORT = "COM7"      # Windows: COM*
ROVER_BAUDRATE = 115200

# Line 37 in main_core.py: Adjust video sources
CCTV_SRC = 0                     # Webcam index
ROVER_SRC = 1                    # Rover camera index
DRONE_SRC = "http://192.168.137.196:8080/?action=stream"  # Drone IP stream
```
Edit `main_core.py` to configure variables:
```python
# ===== DETECTION TOGGLES =====
ENABLE_TRESPASS = False          # Enable/disable trespass detection
ENABLE_ALTERCATION = False       # Enable/disable altercation detection
ENABLE_FACE = True               # Enable/disable face recognition
ENABLE_ATTRIBUTES = False        # Enable/disable attribute extraction

### Step 4: Run the System
```bash
python app.py
```

The web dashboard will be available at: `http://localhost:5000`

Enjoy 😊
