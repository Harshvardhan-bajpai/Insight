from flask import Flask, render_template, Response, jsonify, request, send_from_directory

import threading
import time
import json
import cv2
import os
import serial  # single place for rover serial

import main_core
from rover_tracking_system import get_global_tracker

app = Flask(__name__, template_folder="templates")

# =========================
# SHARED STATE
# =========================

latest_frames = {
    "CCTV": None,
    "ROVER": None,
    "DRONE": None,
}

latest_alerts = []
system_stats = {
    "total_people": 0,
    "alerts": 0,
}

lock = threading.Lock()

# =========================
# ROVER SERIAL (shared)
# =========================

ROVER_SERIAL_PORT = "COM7"   # CHANGE ME
ROVER_BAUDRATE = 115200
_rover_serial = None


def get_rover_serial():
    global _rover_serial
    if _rover_serial is None:
        try:
            _rover_serial = serial.Serial(
                ROVER_SERIAL_PORT, ROVER_BAUDRATE, timeout=0.1
            )
            print(f"[ROVER] Connected to {ROVER_SERIAL_PORT}")
        except Exception as e:
            print(f"[ROVER] Failed to open {ROVER_SERIAL_PORT}: {e}")
            _rover_serial = None
    return _rover_serial


def send_rover_command(cmd: str):
    ser = get_rover_serial()
    if ser is None:
        return
    try:
        ser.write((cmd + "\n").encode("utf-8"))
        print(f"[ROVER] Sent: {cmd}")
    except Exception as e:
        print(f"[ROVER] Serial error: {e}")


# =========================
# BACKGROUND CORE RUNNER
# =========================

def core_runner():
    global latest_alerts, system_stats

    def frame_callback(feed, frame):
        with lock:
            latest_frames[feed] = frame

    def alert_callback(alert):
        with lock:
            latest_alerts.append(alert)
            if len(latest_alerts) > 200:
                latest_alerts.pop(0)
            system_stats["alerts"] += 1

    def stats_callback(people_count):
        with lock:
            system_stats["total_people"] = people_count

    print("[CORE] Starting INSIGHT core system...")
    main_core.run_system(
        frame_callback=frame_callback,
        alert_callback=alert_callback,
        stats_callback=stats_callback,
    )


# =========================
# PAGE ROUTES
# =========================

@app.route("/")
def home():
    return render_template("controlpanel.html")


@app.route("/drone")
def drone_page():
    return render_template("dronepanel.html")


@app.route("/rover")
def rover_page():
    return render_template("roverpanel.html")


@app.route("/cctv")
def cctv_page():
    return render_template("cctvpanel.html")


@app.route("/facedb")
def facedb_page():
    return render_template("facedatabase.html")


@app.route("/events")
def events_page():
    return render_template("events.html")


@app.route("/api/events")
def get_events():
    try:
        with open("events_db.json", "r") as f:
            return jsonify(json.load(f))
    except Exception:
        return jsonify([])


# =========================
# FACE ENROLLMENT API
# =========================

@app.route("/api/enroll_face", methods=["POST"])
def enroll_face_api():
    data = request.get_json() or {}
    name = data.get("name")
    unique_id = data.get("person_id") or data.get("unique_id")
    category = data.get("category")
    img_data = data.get("image")

    if not all([name, unique_id, category, img_data]):
        return jsonify({"error": "missing fields"}), 400

    import base64, re, time as _time

    m = re.match(r"data:(image/\w+);base64,(.*)", img_data)
    if m:
        ext = m.group(1).split("/")[-1]
        b64 = m.group(2)
    else:
        ext = "jpg"
        b64 = img_data

    try:
        img_bytes = base64.b64decode(b64)
    except Exception:
        return jsonify({"error": "invalid image data"}), 400

    faces_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces")
    os.makedirs(faces_dir, exist_ok=True)

    filename = f"{unique_id}_{int(_time.time())}.{ext}"
    image_path = os.path.join(faces_dir, filename)

    try:
        with open(image_path, "wb") as f:
            f.write(img_bytes)
    except Exception:
        return jsonify({"error": "failed to save image"}), 500

    try:
        from face_enrollment import enroll_face
        record = enroll_face(image_path, name, category, unique_id)
    except Exception as e:
        print("[ENROLL ERROR]", repr(e))
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "ok", "record": record})


# =========================
# FACE DB + IMAGES
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_DB_FILE = os.path.join(BASE_DIR, "faces", "face_db.json")
FACES_DIR = os.path.join(BASE_DIR, "faces")


@app.route("/faces/<path:filename>")
def faces_static(filename):
    return send_from_directory(FACES_DIR, filename)


@app.route("/api/faces")
def get_faces():
    if not os.path.exists(FACE_DB_FILE):
        return jsonify([])
    try:
        with open(FACE_DB_FILE, "r") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# ROVER COMMAND API (manual)
# =========================

@app.route("/api/rover/command", methods=["POST"])
def rover_command():
    data = request.get_json() or {}
    cmd = (data.get("command") or "").strip().upper()

    if not cmd or len(cmd) > 2:
        return jsonify({"error": "invalid command"}), 400

    if cmd not in ["W", "A", "S", "D", "G", "H", "F", "X", "T", "WT", "AT", "DT"]:
        return jsonify({"error": "unsupported command"}), 400

    send_rover_command(cmd)
    return jsonify({"status": "ok", "command": cmd})


# =========================
# ROVER TRACKING API (auto)
# =========================

@app.route("/api/rover/switch_target", methods=["POST"])
def rover_switch_target():
    tracker = get_global_tracker()
    if tracker is None:
        return jsonify({"error": "tracker not ready"}), 500
    tracker.switch_active_person()
    return jsonify({"status": "ok"})


@app.route("/api/rover/toggle_lock", methods=["POST"])
def rover_toggle_lock():
    tracker = get_global_tracker()
    if tracker is None:
        return jsonify({"error": "tracker not ready"}), 500
    locked = tracker.toggle_lock()
    return jsonify({"status": "ok", "locked": locked})


# =========================
# VIDEO STREAM ENDPOINTS
# =========================

def generate_stream(feed_name):
    while True:
        try:
            with lock:
                frame = latest_frames.get(feed_name)
            if frame is None:
                time.sleep(0.05)
                continue

            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame_bytes +
                b"\r\n"
            )

            time.sleep(0.03)
        except Exception as e:
            print(f"[STREAM] Error on {feed_name}: {e}")
            time.sleep(0.2)


@app.route("/video/<feed_name>")
def video_feed(feed_name):
    feed_name = feed_name.upper()
    if feed_name not in latest_frames:
        return "Invalid feed", 404

    return Response(
        generate_stream(feed_name),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# =========================
# ALERT + STATS API
# =========================

@app.route("/api/alerts")
def get_alerts():
    with lock:
        return jsonify(latest_alerts[-20:])


@app.route("/api/stats")
def get_stats():
    with lock:
        return jsonify(system_stats)


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    core_thread = threading.Thread(target=core_runner, daemon=True)
    core_thread.start()

    print("[FLASK] INSIGHT Control Panel running at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
