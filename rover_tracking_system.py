# rover_tracking_system.py

import cv2
import time
from ultralytics import YOLO

from rover_face_watch import check_people_watchlist  # <-- you create this helper


class RoverTrackingSystem:
    def __init__(self, command_sender):
        """
        command_sender: function(cmd: str) -> None
        Used to send WT / AT / DT to the rover (via serial).
        """
        self.command_sender = command_sender

        self.model = YOLO("yolov8n.pt")

        self.detected_people = []      # list of (x,y,w,h,pid,conf)
        self.active_person_index = -1  # which box is "selected" (red)
        self.locked_person_index = -1  # which box is locked (blue)
        self.tracking = False

        self.missed_frames = 0
        self.max_missed_frames = 5

        self.movement_threshold = 25   # px offset from center
        self.size_ratio_threshold = 0.4
        self.last_command_time = time.time()
        self.command_cooldown = 0.18

        # recognition control
        self.last_recog_time = 0.0
        self.recog_cooldown = 0.7  # seconds between recognition sweeps

    # -------- core helpers --------

    def _send_command(self, cmd):
        if self.command_sender:
            self.command_sender(cmd)

    def detect_people(self, frame):
        results = self.model(frame, imgsz=320, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            if int(box.cls[0]) == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                boxes.append((x1, y1, x2 - x1, y2 - y1, conf))

        if not boxes:
            self.missed_frames += 1
            if self.missed_frames > self.max_missed_frames:
                self.detected_people = []
                if self.locked_person_index != -1:
                    self.locked_person_index = -1
                    self.tracking = False
            return

        self.missed_frames = 0
        new_people = []
        pid = 1
        for (x, y, w, h, conf) in boxes:
            new_people.append((x, y, w, h, pid, conf))
            pid += 1
        self.detected_people = new_people

        if self.active_person_index >= len(new_people):
            self.active_person_index = -1
        if self.locked_person_index >= len(new_people):
            self.locked_person_index = -1
            self.tracking = False

    def switch_active_person(self):
        if not self.detected_people:
            return
        if self.active_person_index == -1:
            self.active_person_index = 0
        else:
            self.active_person_index = (self.active_person_index + 1) % len(self.detected_people)

    def toggle_lock(self):
        if self.active_person_index == -1 or not self.detected_people:
            return self.tracking
        if self.locked_person_index == -1:
            self.locked_person_index = self.active_person_index
            self.tracking = True
        else:
            self.locked_person_index = -1
            self.tracking = False
        return self.tracking

    def _adjust_position(self, offset_x, offset_y, size_ratio, frame):
        cmd = None
        # left/right
        if abs(offset_x) > self.movement_threshold:
            cmd = "DT" if offset_x > 0 else "AT"
        # forward
        elif size_ratio < self.size_ratio_threshold * 0.27:
            cmd = "WT"

        if cmd:
            self._send_command(cmd)
            cv2.putText(frame, f"Cmd: {cmd}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def _track_locked_person(self, frame):
        if self.locked_person_index == -1 or not self.detected_people:
            return
        if self.locked_person_index >= len(self.detected_people):
            self.locked_person_index = -1
            self.tracking = False
            return

        x, y, w, h, pid, conf = self.detected_people[self.locked_person_index]
        cx, cy = x + w // 2, y + h // 2
        fx, fy = frame.shape[1] // 2, frame.shape[0] // 2
        offset_x, offset_y = cx - fx, cy - fy
        size_ratio = (w * h) / (frame.shape[1] * frame.shape[0])

        # Blue box for locked target
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"Person {pid} (LOCKED)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if time.time() - self.last_command_time > self.command_cooldown:
            self._adjust_position(offset_x, offset_y, size_ratio, frame)
            self.last_command_time = time.time()

    def _draw_interface(self, frame):
        fx, fy = frame.shape[1] // 2, frame.shape[0] // 2
        cv2.line(frame, (fx - 20, fy), (fx + 20, fy), (255, 255, 255), 1)
        cv2.line(frame, (fx, fy - 20), (fx, fy + 20), (255, 255, 255), 1)

        if self.locked_person_index != -1 and self.locked_person_index < len(self.detected_people):
            # locked person already drawn as blue
            return

        for i, (x, y, w, h, pid, _) in enumerate(self.detected_people):
            if i == self.active_person_index:
                color = (0, 0, 255)   # red = active
            else:
                color = (0, 255, 0)   # green = others
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            status = " (ACTIVE)" if i == self.active_person_index else ""
            cv2.putText(frame, f"Person {pid}{status}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def update(self, frame):
        """
        Call this every frame for the ROVER feed.
        Returns the frame with boxes drawn.
        """
        # YOLO detection for all people
        self.detect_people(frame)

        # WATCHLIST RECOGNITION FOR ALL BOXES (only if not locked)
        if not self.tracking or self.locked_person_index == -1:
            now = time.time()
            if now - self.last_recog_time > self.recog_cooldown and self.detected_people:
                self.last_recog_time = now
                idx = check_people_watchlist(frame, self.detected_people)
                if idx != -1:
                    # auto-lock on watchlisted person
                    self.active_person_index = idx
                    self.locked_person_index = idx
                    self.tracking = True
                    cv2.putText(frame, "WATCHLIST MATCH - AUTO LOCK",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Once locked, recognition condition above stops running (tracking only)

        if self.tracking and self.locked_person_index != -1:
            self._track_locked_person(frame)

        self._draw_interface(frame)
        return frame


# Global tracker handle so app.py can access it
global_tracker = None


def set_global_tracker(tracker):
    global global_tracker
    global_tracker = tracker


def get_global_tracker():
    return global_tracker
