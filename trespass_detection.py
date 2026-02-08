# trespass_detection.py

# Restricted zones (manually define per camera later)
# Format: [(x1, y1, x2, y2), ...]
RESTRICTED_ZONES = [
    (100, 50, 300, 250),   # Example zone 1
    (400, 100, 600, 300)  # Example zone 2
]


def point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def detect_trespass(people):
    events = []

    for p in people:
        cx, cy = p["center"]

        for zone in RESTRICTED_ZONES:
            if point_in_box((cx, cy), zone):
                events.append({
                    "person_id": p["id"],
                    "zone": zone,
                    "type": "trespass"
                })
                break

    return events
