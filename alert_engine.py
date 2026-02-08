# alert_engine.py
import time
import json
import os
import threading

# =========================
# COOLDOWN SETTINGS (seconds)
# =========================

COOLDOWNS = {
    "altercation_flag": 2.0,
    "altercation_locked": 2.5,
    "trespass": 8.0,
    "face_unknown": 4.0,
    "face_match": 6.0
}

# =========================
# INTERNAL STATE
# =========================

_last_event_times = {}
_db_lock = threading.Lock()

# =========================
# EVENT JSON DATABASE
# =========================

EVENT_DB_FILE = "events_db.json"

# Ensure DB exists and is valid
if not os.path.exists(EVENT_DB_FILE):
    with open(EVENT_DB_FILE, "w") as f:
        json.dump([], f)


def _safe_load_db():
    """Load JSON safely even if file was corrupted or empty."""
    try:
        with open(EVENT_DB_FILE, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def save_event_to_db(alert):
    """Thread-safe persistent storage."""
    with _db_lock:
        data = _safe_load_db()
        data.append(alert)

        # hard cap to prevent infinite growth
        if len(data) > 3000:
            data = data[-3000:]

        with open(EVENT_DB_FILE, "w") as f:
            json.dump(data, f, indent=2)


# =========================
# COOLDOWN UTILITY
# =========================

def _on_cooldown(event_type, key):
    now = time.time()
    cooldown = COOLDOWNS.get(event_type, 5)

    last = _last_event_times.get(key)
    if last and now - last < cooldown:
        return True

    _last_event_times[key] = now
    return False


# =========================
# ATTRIBUTE CACHE (FOR UI DESCRIPTION)
# =========================

ATTRIBUTE_CACHE = {}
ATTRIBUTE_CACHE_TTL = 2.0
ATTRIBUTE_MIN_SAMPLES = 4


def _update_attribute_cache(attributes):
    now = time.time()

    for a in attributes:
        pid = a.get("person_id")
        if pid is None:
            continue

        entry = ATTRIBUTE_CACHE.get(pid, {
            "gender": [],
            "upper": [],
            "lower": [],
            "last_seen": now
        })

        if a.get("gender"):
            entry["gender"].append(a["gender"])
        if a.get("upper_color"):
            entry["upper"].append(a["upper_color"])
        if a.get("lower_color"):
            entry["lower"].append(a["lower_color"])

        entry["last_seen"] = now
        ATTRIBUTE_CACHE[pid] = entry

    # cleanup stale
    for pid in list(ATTRIBUTE_CACHE.keys()):
        if now - ATTRIBUTE_CACHE[pid]["last_seen"] > ATTRIBUTE_CACHE_TTL:
            del ATTRIBUTE_CACHE[pid]


def _describe_person(pid):
    entry = ATTRIBUTE_CACHE.get(pid)
    if not entry:
        return "Person involved"

    if len(entry["gender"]) < ATTRIBUTE_MIN_SAMPLES:
        return "Person involved"

    gender = max(set(entry["gender"]), key=entry["gender"].count)

    upper = (
        max(set(entry["upper"]), key=entry["upper"].count)
        if entry["upper"] else None
    )
    lower = (
        max(set(entry["lower"]), key=entry["lower"].count)
        if entry["lower"] else None
    )

    if upper and lower:
        return f"{gender.capitalize()}, {upper} and {lower} clothes"
    if upper:
        return f"{gender.capitalize()}, {upper} clothes"

    return gender.capitalize()


# =========================
# MAIN ENTRY POINT
# =========================

def process_events(detections):
    """
    Expected detections format from main_core.py:
    {
        "feed": "CCTV",
        "altercations": [{
            "flagged_ids": [...],
            "locked_id": int | None,
            "confidence": {id: float}
        }],
        "trespass": [...],
        "faces": [...],
        "attributes": [...]
    }
    """

    alerts = []
    feed = detections.get("feed", "UNKNOWN")

    # update attribute cache FIRST
    _update_attribute_cache(detections.get("attributes", []))

    # =========================
    # ALTERCATION EVENTS
    # =========================

    for alt in detections.get("altercations", []):
        flagged_ids = alt.get("flagged_ids", [])
        confidence_map = alt.get("confidence", {})
        locked_id = alt.get("locked_id")

        # --- Possible altercation ---
        for pid in flagged_ids:
            key = f"alt_flag_{pid}"

            if _on_cooldown("altercation_flag", key):
                continue

            alert = {
                "timestamp": time.time(),
                "type": "altercation",
                "title": "Possible altercation detected",
                "description": _describe_person(pid),
                "feed": feed,
                "confidence": round(confidence_map.get(pid, 0.75), 2),
                "severity": "Medium"
            }

            alerts.append(alert)
            save_event_to_db(alert)

        # --- Confirmed altercation ---
        if locked_id is not None:
            key = f"alt_lock_{locked_id}"

            if not _on_cooldown("altercation_locked", key):
                alert = {
                    "timestamp": time.time(),
                    "type": "altercation",
                    "title": "Altercation detected",
                    "description": _describe_person(locked_id),
                    "feed": feed,
                    "confidence": round(
                        confidence_map.get(locked_id, 0.85), 2
                    ),
                    "severity": "High"
                }

                alerts.append(alert)
                save_event_to_db(alert)

    # =========================
    # TRESPASS EVENTS
    # =========================

    for ev in detections.get("trespass", []):
        pid = ev.get("person_id", -1)
        key = f"trespass_{pid}"

        if _on_cooldown("trespass", key):
            continue

        alert = {
            "timestamp": time.time(),
            "type": "trespass",
            "title": "Trespass detected",
            "description": _describe_person(pid),
            "feed": feed,
            "confidence": round(ev.get("confidence", 0.9), 2),
            "severity": "Medium"
        }

        alerts.append(alert)
        save_event_to_db(alert)

    # =========================
    # FACE RECOGNITION EVENTS
    # =========================

    for face in detections.get("faces", []):
        name = face.get("name", "UNKNOWN")
        category = face.get("category", "UNKNOWN")
        conf = round(face.get("confidence", 0), 2)

        # --- Unknown face ---
        if category == "UNKNOWN":
            key = f"face_unknown_{feed}"

            if _on_cooldown("face_unknown", key):
                continue

            alert = {
                "timestamp": time.time(),
                "type": "face_unknown",
                "title": "Unknown face detected",
                "description": "Unidentified individual",
                "feed": feed,
                "confidence": conf,
                "severity": "Low"
            }

            alerts.append(alert)
            save_event_to_db(alert)
            continue

        # --- Watchlist / Ex-convict ---
        if category in ["Criminal", "Watchlist", "Ex-Convict"]:
            key = f"face_watch_{name}"
            if _on_cooldown("face_match", key):
                continue

            alert_type = "face_criminal" if category == "Criminal" else "face_watchlist"

            alert = {
                "timestamp": time.time(),
                "type": alert_type,
                "title": f"{category} detected",
                "description": name,
                "feed": feed,
                "confidence": conf,
                "severity": "Medium"
            }

            alerts.append(alert)
            save_event_to_db(alert)
            continue


        # --- Known safe person ---
        key = f"face_known_{name}"

        if _on_cooldown("face_match", key):
            continue

        alert = {
            "timestamp": time.time(),
            "type": "face_known",
            "title": f"{category} detected",
            "description": name,
            "feed": feed,
            "confidence": conf,
            "severity": "Low"
        }

        alerts.append(alert)
        save_event_to_db(alert)

    return alerts
