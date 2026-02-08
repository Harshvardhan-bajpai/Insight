import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ================= CONFIG ================= #

MAX_HISTORY = 30
LOCK_DURATION_SECONDS = 7

# Altercation logic thresholds
STRIKE_SPEED_THRESHOLD = 8
CONTACT_DISTANCE = 120
REACTION_THRESHOLD = 25
DIRECTION_SIM_THRESHOLD = 0.3

# ================= INIT ================= #

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("sample/punch.mp4")

track_history = {}
bbox_history = {}

next_track_id = 1

locked_target_id = None
lock_start_time = None

# ================= UTILS ================= #

def centroid(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def dist(a, b):
    return np.linalg.norm(a - b)

def cosine_sim(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def boxes_touch(a, b, margin=15):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (
        ax2 + margin < bx1 or ax1 - margin > bx2 or
        ay2 + margin < by1 or ay1 - margin > by2
    )

# ================= MAIN LOOP ================= #

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    results = model(frame, classes=[0], conf=0.4)
    current_tracks = {}

    # ---------- PERSON TRACKING ---------- #
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            c = centroid((x1, y1, x2, y2))

            matched = None
            min_d = 9999

            for tid, hist in track_history.items():
                if not hist:
                    continue
                d = dist(hist[-1], c)
                if d < 60 and d < min_d:
                    min_d = d
                    matched = tid

            if matched is None:
                matched = next_track_id
                next_track_id += 1
                track_history[matched] = deque(maxlen=MAX_HISTORY)
                bbox_history[matched] = deque(maxlen=MAX_HISTORY)

            track_history[matched].append(c)
            bbox_history[matched].append((x1, y1, x2, y2))
            current_tracks[matched] = (x1, y1, x2, y2)

    # ---------- ALTERCATION DETECTION ---------- #
    flagged = set()
    confidence = {}

    for attacker_id in track_history:
        if len(track_history[attacker_id]) < 2:
            continue

        a_now = track_history[attacker_id][-1]
        a_prev = track_history[attacker_id][-2]
        a_speed = dist(a_now, a_prev)
        a_vec = a_now - a_prev

        if a_speed < STRIKE_SPEED_THRESHOLD:
            continue

        for victim_id in track_history:
            if attacker_id == victim_id:
                continue
            if len(track_history[victim_id]) < 5:
                continue

            v_now = track_history[victim_id][-1]
            v_prev = track_history[victim_id][-5]
            v_disp = v_now - v_prev

            if dist(a_now, v_now) > CONTACT_DISTANCE:
                continue

            if not boxes_touch(
                bbox_history[attacker_id][-1],
                bbox_history[victim_id][-1]
            ):
                continue

            reaction_mag = np.linalg.norm(v_disp)
            if reaction_mag < REACTION_THRESHOLD:
                continue

            dir_sim = cosine_sim(a_vec, v_disp)
            if dir_sim < DIRECTION_SIM_THRESHOLD:
                continue

            conf = min(
                1.0,
                0.5 * (a_speed / 10) +
                0.5 * (reaction_mag / 40)
            )

            flagged.add(attacker_id)
            flagged.add(victim_id)
            confidence[attacker_id] = conf
            confidence[victim_id] = conf

    # ---------- TARGET LOCK ---------- #
    if locked_target_id is not None:
        if current_time - lock_start_time > LOCK_DURATION_SECONDS:
            locked_target_id = None
            lock_start_time = None

    if locked_target_id is None and flagged:
        locked_target_id = list(flagged)[0]
        lock_start_time = current_time

    # ---------- DRAW ---------- #
    for tid, (x1, y1, x2, y2) in current_tracks.items():
        color = (0, 255, 0)
        label = f"ID {tid}"

        if tid in flagged:
            color = (0, 0, 255)

        if tid == locked_target_id:
            color = (255, 0, 0)
            remaining = LOCK_DURATION_SECONDS - int(current_time - lock_start_time)
            label += f" LOCK {remaining}s"

        if tid in confidence:
            label += f" {int(confidence[tid] * 100)}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    cv2.imshow("INSIGHT Altercation Detection", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
