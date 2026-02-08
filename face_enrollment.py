import os
import json
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from datetime import datetime

# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_DB_DIR = os.path.join(BASE_DIR, "faces")
EMBEDDINGS_PATH = os.path.join(FACE_DB_DIR, "embeddings.npy")
META_PATH = os.path.join(FACE_DB_DIR, "face_db.json")
os.makedirs(FACE_DB_DIR, exist_ok=True)

# =========================
# INIT INSIGHTFACE (CPU)
# =========================

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
# Slightly smaller det_size for speed; tweak if you want
app.prepare(ctx_id=0, det_size=(480, 480))

# =========================
# LOAD / SAVE HELPERS
# =========================

def _load_embeddings():
    if os.path.exists(EMBEDDINGS_PATH):
        try:
            return np.load(EMBEDDINGS_PATH)
        except Exception:
            return np.empty((0, 512), dtype=np.float32)
    return np.empty((0, 512), dtype=np.float32)

def _save_embeddings(embeddings):
    tmp_path = EMBEDDINGS_PATH + ".tmp.npy"
    np.save(tmp_path, embeddings)
    os.replace(tmp_path, EMBEDDINGS_PATH)

def _load_metadata():
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def _save_metadata(data):
    tmp_path = META_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, META_PATH)

# =========================
# MAIN ENROLL FUNCTION
# =========================

def enroll_face(
    image_path: str,
    name: str,
    category: str,
    unique_id: str
):
    """
    Enrolls a face into the system.
    category:
    Criminal | Watchlist | Ex-Convict | Employee | Resident | Security
    """
    if not os.path.exists(image_path):
        raise ValueError("Image path does not exist")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to read image")

    faces = app.get(img)
    if len(faces) == 0:
        raise ValueError("No face detected")
    if len(faces) > 1:
        raise ValueError("Multiple faces detected — upload a single-face image")

    face = faces[0]
    if not hasattr(face, "embedding"):
        raise ValueError("Failed to compute face embedding")

    embedding = face.embedding.astype(np.float32)

    # =========================
    # LOAD EXISTING DATABASE
    # =========================

    embeddings_db = _load_embeddings()
    metadata_db = _load_metadata()

    # =========================
    # DUPLICATE FACE CHECK
    # =========================

    if embeddings_db.shape[0] > 0:
        sims = np.dot(embeddings_db, embedding)
        if np.max(sims) > 0.85:
            raise ValueError("Face already exists in database")

    # =========================
    # UNIQUE ID CHECK
    # =========================

    if any(m.get("unique_id") == unique_id for m in metadata_db):
        raise ValueError("unique_id already exists in database")

    # =========================
    # SAVE IMAGE
    # =========================

    image_filename = f"{unique_id}.jpg"
    image_save_path = os.path.join(FACE_DB_DIR, image_filename)
    cv2.imwrite(image_save_path, img)

    # =========================
    # SAVE EMBEDDING
    # =========================

    if embeddings_db.size == 0:
        embeddings_db = embedding.reshape(1, -1)
    else:
        embeddings_db = np.vstack([embeddings_db, embedding])
    _save_embeddings(embeddings_db)

    # =========================
    # SAVE METADATA
    # =========================

    record = {
        "db_index": len(metadata_db),
        "unique_id": unique_id,
        "name": name,
        "category": category,
        # REQUIRED FOR UI + RECOGNITION
        "image": f"/faces/{image_filename}",
        "embedding": f"{unique_id}.npy",
        "enrolled_at": datetime.utcnow().isoformat()
    }

    metadata_db.append(record)
    _save_metadata(metadata_db)

    return record
