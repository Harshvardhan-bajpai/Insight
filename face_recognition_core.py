# face_recognition_core.py

import numpy as np
import json
import os
import threading
from insightface.app import FaceAnalysis

# =========================
# THREAD-LOCAL MODEL
# =========================

_thread_local = threading.local()

def _get_face_app():
    if not hasattr(_thread_local, "app"):
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        app.prepare(ctx_id=0, det_size=(480, 480))
        _thread_local.app = app
    return _thread_local.app

# =========================
# DATABASE FILES
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(BASE_DIR, "faces")
FACE_JSON = os.path.join(FACES_DIR, "face_db.json")
EMB_FILE = os.path.join(FACES_DIR, "embeddings.npy")
os.makedirs(FACES_DIR, exist_ok=True)

# =========================
# SAFE DATABASE LOAD
# =========================

def _load_face_db():
    """
    Loads metadata + embeddings safely.
    NEVER crashes even if files are empty or corrupted.
    """
    # ---- Metadata ----
    if not os.path.exists(FACE_JSON):
        meta = []
    else:
        try:
            with open(FACE_JSON, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = []

    # ---- Embeddings ----
    if not os.path.exists(EMB_FILE):
        embeddings = np.empty((0, 512), dtype=np.float32)
    else:
        try:
            embeddings = np.load(EMB_FILE)
            if embeddings.ndim != 2 or embeddings.size == 0:
                raise ValueError("Invalid embedding file")
        except Exception:
            # HARD RECOVERY — overwrite corrupted file
            embeddings = np.empty((0, 512), dtype=np.float32)
            np.save(EMB_FILE, embeddings)

    # ---- Sync safety ----
    n = min(len(meta), len(embeddings))
    meta = meta[:n]
    embeddings = embeddings[:n]

    return meta, embeddings

# =========================
# SIMILARITY
# =========================

def cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# =========================
# MAIN RECOGNITION
# =========================

MATCH_THRESHOLD = 0.34  # good for demo, tighten later

def recognize_faces(face_detections):
    """
    Input:
        [{ bbox, crop }]
    Output:
        [{
            bbox,
            name,
            category,
            person_id,
            confidence
        }]
    """
    results = []
    if not face_detections:
        return results

    app = _get_face_app()
    meta_db, embeddings_db = _load_face_db()

    for item in face_detections:
        crop = item.get("crop")
        bbox = item.get("bbox")
        if crop is None:
            continue

        faces = app.get(crop)
        if not faces:
            continue

        emb = faces[0].embedding

        # ---- No DB yet ----
        if embeddings_db.shape[0] == 0:
            results.append({
                "bbox": bbox,
                "name": "UNKNOWN",
                "category": "UNKNOWN",
                "person_id": None,
                "confidence": 0.0
            })
            continue

        sims = np.dot(embeddings_db, emb) / (
            np.linalg.norm(embeddings_db, axis=1) * np.linalg.norm(emb) + 1e-6
        )

        idx = int(np.argmax(sims))
        score = float(sims[idx])

        if score < MATCH_THRESHOLD:
            results.append({
                "bbox": bbox,
                "name": "UNKNOWN",
                "category": "UNKNOWN",
                "person_id": None,
                "confidence": score
            })
            continue

        matched = meta_db[idx]
        results.append({
            "bbox": bbox,
            "name": matched.get("name", "UNKNOWN"),
            "category": matched.get("category", "UNKNOWN"),
            "person_id": matched.get("unique_id"),
            "confidence": score
        })

    return results
