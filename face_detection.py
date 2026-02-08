import cv2
import os
import numpy as np  # if not already imported elsewhere

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_detection_yunet_2023mar.onnx")

# Use a bit larger default input size for better small / angled faces
INPUT_W, INPUT_H = 416, 416

_face_detector = None

def _get_detector():
    global _face_detector
    if _face_detector is None:
        _face_detector = cv2.FaceDetectorYN.create(
            model=MODEL_PATH,
            config="",
            input_size=(INPUT_W, INPUT_H),
            # more permissive for off-angle / low-contrast faces
            score_threshold=0.2,
            nms_threshold=0.3,
            top_k=500,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )
    return _face_detector


def detect_faces(frame):
    """
    Returns: list of dicts {bbox: [x1,y1,x2,y2], crop: np.ndarray}
    """
    detector = _get_detector()

    h, w = frame.shape[:2]

    # If you are downscaling frames in main_core, avoid making them too small.
    # You want face height >= ~80 px in this 'frame'.

    detector.setInputSize((w, h))

    # YuNet expects BGR uint8; ensure correct type
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    # YuNet returns: [x, y, w, h, score, ... landmarks ...]
    _, faces = detector.detect(frame)

    results = []
    if faces is None:
        return results

    for f in faces:
        x, y, w_box, h_box, score = f[:5]

        # You can add an extra score filter if needed
        if score < 0.2:
            continue

        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(w, int(x + w_box))
        y2 = min(h, int(y + h_box))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        results.append({
            "bbox": [x1, y1, x2, y2],
            "crop": crop
        })

    return results
