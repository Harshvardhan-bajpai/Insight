import cv2
import numpy as np
import mediapipe as mp
import threading

# =========================
# THREAD-LOCAL STATE
# =========================
_thread_local = threading.local()

def _get_pose():
    if not hasattr(_thread_local, "pose"):
        mp_pose = mp.solutions.pose
        _thread_local.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return _thread_local.pose


def _get_prev_keypoints():
    if not hasattr(_thread_local, "prev_keypoints"):
        _thread_local.prev_keypoints = None
    return _thread_local.prev_keypoints


def _set_prev_keypoints(kp):
    _thread_local.prev_keypoints = kp


def _get_wave_count():
    if not hasattr(_thread_local, "wave_count"):
        _thread_local.wave_count = 0
    return _thread_local.wave_count


def _set_wave_count(v):
    _thread_local.wave_count = v


# =========================
# HYPERPARAMETERS (UNCHANGED)
# =========================
HAND_ABOVE_SHOULDER_OFFSET = 15
WAVE_SPEED_THRESHOLD = 12
MIN_WAVE_FRAMES = 6


# =========================
# ORIGINAL FUNCTIONS (UNCHANGED)
# =========================

def extract_pose_landmarks(frame):
    pose = _get_pose()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if not results.pose_landmarks:
        return None

    h, w, _ = frame.shape
    landmarks = {}

    for idx, lm in enumerate(results.pose_landmarks.landmark):
        landmarks[idx] = (int(lm.x * w), int(lm.y * h))

    return landmarks


def get_keypoints(landmarks):
    return {
        "left_wrist": landmarks.get(15),
        "right_wrist": landmarks.get(16),
        "left_shoulder": landmarks.get(11),
        "right_shoulder": landmarks.get(12)
    }


def velocity(p1, p2):
    if p1 is None or p2 is None:
        return 0
    return np.linalg.norm(np.array(p2) - np.array(p1))


# =========================
# THREAD-SAFE DETECTOR
# =========================

def detect_waving(frame, people):
    events = []

    landmarks = extract_pose_landmarks(frame)
    if landmarks is None:
        return events

    keypoints = get_keypoints(landmarks)

    prev = _get_prev_keypoints()
    if prev is None:
        _set_prev_keypoints(keypoints)
        _set_wave_count(0)
        return events

    _set_prev_keypoints(keypoints)

    # Extract points
    lw = keypoints["left_wrist"]
    rw = keypoints["right_wrist"]
    ls = keypoints["left_shoulder"]
    rs = keypoints["right_shoulder"]

    if not lw or not rw or not ls or not rs:
        _set_wave_count(0)
        return events

    # Check hands above shoulders
    hands_up = (
        lw[1] < ls[1] - HAND_ABOVE_SHOULDER_OFFSET and
        rw[1] < rs[1] - HAND_ABOVE_SHOULDER_OFFSET
    )

    if not hands_up:
        _set_wave_count(0)
        return events

    # Wrist velocities
    v_l = velocity(prev.get("left_wrist"), lw)
    v_r = velocity(prev.get("right_wrist"), rw)

    wave_count = _get_wave_count()

    # Check oscillation
    if v_l > WAVE_SPEED_THRESHOLD or v_r > WAVE_SPEED_THRESHOLD:
        wave_count += 1
    else:
        wave_count = max(0, wave_count - 1)

    if wave_count >= MIN_WAVE_FRAMES:
        wave_count = 0
        events.append({
            "type": "waving",
            "confidence": 0.8
        })

    _set_wave_count(wave_count)

    return events
