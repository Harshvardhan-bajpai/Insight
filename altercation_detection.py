# altercation_detection.py
import cv2
import numpy as np
from collections import deque
import time
import os

from attributes import get_attributes  # <-- IMPORTANT

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =========================
# CONFIG
# =========================

FAST_MOVE_THRESHOLD = 8
PUSH_SPEED_THRESHOLD = 5
CONTACT_DISTANCE = 120
REACTION_THRESHOLD = 25
DIRECTION_SIM_THRESHOLD = 0.3

MAX_HISTORY = 30
LOCK_DURATION_SECONDS = 2.5

# =========================
# INTERNAL STATE
# =========================

track_history = {}
bbox_history = {}

locked_target_id = None
lock_start_time = None

# =========================
# UTILS
# =========================

def compute_centroid(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)


def cosine_similarity(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def boxes_touch(boxA, boxB, margin=15):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    return not (
        ax2 + margin < bx1 or
        ax1 - margin > bx2 or
        ay2 + margin < by1 or
        ay1 - margin > by2
    )


def build_description(frame, bbox, person_id):
    """
    Extracts attributes ONCE and builds a stable description string.
    """

    attr = get_attributes(frame, bbox, None)

    gender = attr.get("gender", "person")

    upper = attr.get("upper_color")
    lower = attr.get("lower_color")

    if upper and lower:
        clothes = f"{upper} and {lower}"
    elif upper:
        clothes = upper
    elif lower:
        clothes = lower
    else:
        clothes = "unknown"

    return f"{gender.capitalize()}, {clothes} clothes"


# =========================
# MAIN FUNCTION
# =========================

def detect_altercation(frame, people):
    """
    Returns a LIST of altercation events.
    Each event already contains description & confidence.
    """

    global locked_target_id, lock_start_time

    current_time = time.time()

    confidence_scores = {}
    descriptions = {}
    final_flags = set()

    # -------------------------
    # UPDATE TRACKS
    # -------------------------
    for p in people:
        pid = p["id"]
        x1, y1, x2, y2 = p["bbox"]

        centroid = compute_centroid((x1, y1, x2, y2))

        if pid not in track_history:
            track_history[pid] = deque(maxlen=MAX_HISTORY)
            bbox_history[pid] = deque(maxlen=MAX_HISTORY)

        track_history[pid].append(centroid)
        bbox_history[pid].append((x1, y1, x2, y2))

    # -------------------------
    # ALTERCATION LOGIC (UNCHANGED)
    # -------------------------
    for attacker_id in track_history:
        if len(track_history[attacker_id]) < 2:
            continue

        attacker_now = track_history[attacker_id][-1]
        attacker_prev = track_history[attacker_id][-2]
        attacker_speed = euclidean(attacker_now, attacker_prev)
        strike_vector = attacker_now - attacker_prev

        if attacker_speed < PUSH_SPEED_THRESHOLD:
            continue

        for victim_id in track_history:
            if victim_id == attacker_id:
                continue

            if len(track_history[victim_id]) < 5:
                continue

            victim_now = track_history[victim_id][-1]
            victim_prev = track_history[victim_id][-5]
            victim_displacement = victim_now - victim_prev
            reaction_mag = np.linalg.norm(victim_displacement)

            dist = euclidean(attacker_now, victim_now)

            if dist > CONTACT_DISTANCE:
                continue

            if not boxes_touch(
                bbox_history[attacker_id][-1],
                bbox_history[victim_id][-1]
            ):
                continue

            reaction_dir_sim = cosine_similarity(
                strike_vector, victim_displacement
            )

            if reaction_mag < REACTION_THRESHOLD:
                continue

            if reaction_dir_sim < DIRECTION_SIM_THRESHOLD:
                continue

            # ---- CONFIDENCE ----
            speed_score = min(attacker_speed / FAST_MOVE_THRESHOLD, 1.0)
            reaction_score = min(reaction_mag / REACTION_THRESHOLD, 1.0)
            direction_score = max(reaction_dir_sim, 0)

            confidence = (
                0.4 * speed_score +
                0.4 * reaction_score +
                0.2 * direction_score
            )

            final_flags.add(attacker_id)
            final_flags.add(victim_id)

            confidence_scores[attacker_id] = confidence
            confidence_scores[victim_id] = confidence

    # -------------------------
    # ATTRIBUTE EXTRACTION (NEW)
    # -------------------------
    for pid in final_flags:
        if pid not in descriptions:
            bbox = bbox_history[pid][-1]
            descriptions[pid] = build_description(frame, bbox, pid)

    # -------------------------
    # LOCK LOGIC (UNCHANGED)
    # -------------------------
    if locked_target_id is not None:
        if current_time - lock_start_time > LOCK_DURATION_SECONDS:
            locked_target_id = None
            lock_start_time = None

    if locked_target_id is None and final_flags:
        locked_target_id = list(final_flags)[0]
        lock_start_time = current_time

    # -------------------------
    # RETURN (FINAL STRUCTURE)
    # -------------------------
    return [{
        "flagged_ids": list(final_flags),
        "locked_id": locked_target_id,
        "confidence": confidence_scores,
        "descriptions": descriptions
    }]
