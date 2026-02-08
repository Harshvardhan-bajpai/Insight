# rover_face_watch.py

from face_detection import detect_faces
from face_recognition_core import recognize_faces

# Use names / IDs that exist in your face_db.json
WATCHLIST_NAMES = {"hru"}  # edit this set


def check_people_watchlist(frame, people):
    """
    frame: full rover frame (BGR)
    people: list of (x, y, w, h, pid, conf) from RoverTrackingSystem.detect_people

    Returns:
        index_in_people (int) of first watchlisted person, or -1 if none.
    """
    if not people:
        return -1

    h_f, w_f = frame.shape[:2]

    for idx, (x, y, w, h, pid, conf) in enumerate(people):
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_f - 1, x + w), min(h_f - 1, y + h)
        if x2 <= x1 or y2 <= y1:
            continue

        person_img = frame[y1:y2, x1:x2]

        faces = detect_faces(person_img)
        if not faces:
            continue

        results = recognize_faces(faces)
        for r in results:
            name = r.get("name") or r.get("person_id")
            if name in WATCHLIST_NAMES:
                return idx

    return -1
