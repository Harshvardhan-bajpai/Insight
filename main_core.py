import cv2
import threading
import queue
import time

from person_detection import detect_and_track
from trespass_detection import detect_trespass
from altercation_detection import detect_altercation
from face_detection import detect_faces
from face_recognition_core import recognize_faces
from attributes import get_attributes
from alert_engine import process_events

from rover_tracking_system import RoverTrackingSystem, set_global_tracker
from app import send_rover_command 

#--------------------------------------------

ENABLE_TRESPASS = False
ENABLE_ALTERCATION = False
ENABLE_FACE = True
ENABLE_ATTRIBUTES = False


CCTV_SRC = 0 
ROVER_SRC = 1   
DRONE_SRC = "http://192.168.137.196:8080/?action=stream"     #"sample/stampede.mp4"

# =========================
# SHARED EVENT QUEUE
# =========================

event_queue = queue.Queue()


def run_system(frame_callback=None, alert_callback=None, stats_callback=None):
    """
    Core runner for INSIGHT system.
    """

    def process_feed(feed_name, src, do_processing=True):
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open {feed_name} feed")
            return

        print(f"[INFO] {feed_name} feed started")

        frame_idx = 0
        SCALE = 0.55

        # Only ROVER uses tracking
        tracker = None
        if feed_name == "ROVER":
            print("[ROVER] Initializing RoverTrackingSystem")
            tracker = RoverTrackingSystem(command_sender=send_rover_command)
            set_global_tracker(tracker)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Downscale for processing
            proc_frame = cv2.resize(
                frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR
            )

            detections = {
                "feed": feed_name,
                "people": [],
                "trespass": [],
                "faces": [],
                "attributes": [],
                "altercations": []
            }

            # Global analytics (CCTV/DRONE); ROVER high‑level analytics are off
            if do_processing and feed_name != "ROVER":
                N = 3
                people = []
                if frame_idx % N == 0:
                    # 1) Person detection + tracking
                    people = detect_and_track(proc_frame)
                    detections["people"] = people

                    # 2) Trespass detection
                    if ENABLE_TRESPASS:
                        detections["trespass"] = detect_trespass(people)

                    # 3) Altercation detection
                    if ENABLE_ALTERCATION:
                        alt = detect_altercation(proc_frame, people)
                        if isinstance(alt, list):
                            detections["altercations"] = alt
                        elif isinstance(alt, dict):
                            detections["altercations"] = [alt] if alt.get("flagged_ids") else []
                        else:
                            detections["altercations"] = []

                    # 4) Face detection + recognition
                    if ENABLE_FACE:
                        faces = detect_faces(proc_frame)
                        detections["faces"] = recognize_faces(faces)

                    # 5) Attributes
                    if ENABLE_ATTRIBUTES:
                        attrs = []
                        for p in people:
                            attr = get_attributes(proc_frame, p["bbox"], None)
                            attr["person_id"] = p["id"]
                            attrs.append(attr)
                        detections["attributes"] = attrs

                    event_queue.put(detections)

            # Auto tracking overlay only for ROVER (always on)
            if feed_name == "ROVER" and tracker is not None:
                # print once in a while if you want:
                # if frame_idx % 30 == 0:
                #     print("[ROVER] calling tracker.update")
                tracked = tracker.update(proc_frame)
                proc_frame = tracked

            # For streaming, send processed frame scaled back up
            frame_for_stream = cv2.resize(proc_frame, (frame.shape[1], frame.shape[0]))

            if frame_callback:
                frame_callback(feed_name, frame_for_stream)

        cap.release()
        print(f"[INFO] {feed_name} feed stopped")

    # =========================
    # ALERT LOOP
    # =========================

    def alert_loop():
        print("[INFO] Alert engine started")
        while True:
            detections = event_queue.get()
            alerts = process_events(detections)
            for alert in alerts:
                if alert_callback:
                    alert_callback(alert)
            if stats_callback:
                stats_callback(len(detections.get("people", [])))

    # =========================
    # THREADS
    # =========================

    threads = [
        threading.Thread(
            target=process_feed,
            args=("CCTV", CCTV_SRC, False),
            daemon=True
        ),
        threading.Thread(
            target=process_feed,
            args=("ROVER", ROVER_SRC, False), 
            daemon=True
        ),
        threading.Thread(
            target=process_feed,
            args=("DRONE", DRONE_SRC, False),
            daemon=True
        ),
        threading.Thread(
            target=alert_loop,
            daemon=True
        )
    ]

    for t in threads:
        t.start()

    while True:
        time.sleep(1)

