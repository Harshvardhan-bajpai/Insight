import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (person class)
model = YOLO("yolov8n.pt")  # you can upgrade to yolov8s.pt later

# Tracker memory
tracks = {}
next_id = 0

# Hyperparameters (you WILL tune these later)
IOU_THRESHOLD = 0.3
MAX_MISSED_FRAMES = 15

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = boxAArea + boxBArea - interArea
    if union == 0:
        return 0

    return interArea / union


def detect_and_track(frame):
    global next_id

    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # YOLO class 0 = person
        if cls == 0 and conf > 0.4:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf
            })

    updated_tracks = {}
    used_detections = set()

    # Match existing tracks with new detections
    for track_id, track in tracks.items():
        best_iou = 0
        best_det_idx = -1

        for i, det in enumerate(detections):
            if i in used_detections:
                continue

            iou_score = iou(track["bbox"], det["bbox"])
            if iou_score > best_iou:
                best_iou = iou_score
                best_det_idx = i

        if best_iou > IOU_THRESHOLD and best_det_idx != -1:
            det = detections[best_det_idx]
            updated_tracks[track_id] = {
                "bbox": det["bbox"],
                "missed": 0
            }
            used_detections.add(best_det_idx)
        else:
            # No match → increase missed counter
            track["missed"] += 1
            if track["missed"] < MAX_MISSED_FRAMES:
                updated_tracks[track_id] = track

    # Add new tracks for unmatched detections
    for i, det in enumerate(detections):
        if i not in used_detections:
            updated_tracks[next_id] = {
                "bbox": det["bbox"],
                "missed": 0
            }
            next_id += 1

    # Replace old tracks
    tracks.clear()
    tracks.update(updated_tracks)

    # Output clean list
    people = []
    for track_id, track in tracks.items():
        x1, y1, x2, y2 = track["bbox"]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        people.append({
            "id": track_id,
            "bbox": [x1, y1, x2, y2],
            "center": [cx, cy]
        })

    return people
