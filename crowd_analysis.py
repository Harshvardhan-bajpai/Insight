# crowd_analysis.py
import cv2
import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class CrowdAlert:
    alert_type: str
    severity: str  # "LOW", "MEDIUM", "CRITICAL", "STAMPEDE"
    confidence: float
    timestamp: float
    countdown: Optional[int] = None  # seconds until predicted stampede
    density: float = 0.0
    panic_score: float = 0.0

class CrowdAnalyzer:
    def __init__(self, frame_width=640, frame_height=480, analysis_zone=None):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.analysis_zone = analysis_zone  # (x1,y1,x2,y2) or None for full frame
        
        # Crowd density thresholds (people per sq meter)
        self.DENSITY_NORMAL = 0.5
        self.DENSITY_WARNING = 2.0
        self.DENSITY_CRITICAL = 4.0
        self.DENSITY_STAMPEDE = 6.0
        
        # Panic detection thresholds
        self.SPEED_THRESHOLD = 2.5  # px/frame
        self.DIRECTION_VARIANCE = 45  # degrees
        self.PANIC_SCORE_THRESHOLD = 0.75
        
        # Tracking buffers (last 30 frames)
        self.centroids_history = deque(maxlen=30)
        self.speeds_history = deque(maxlen=30)
        self.directions_history = deque(maxlen=30)
        
        self.last_frame_time = time.time()
        self.frame_count = 0
        
    def update_centroids(self, people: List[Dict]) -> List[Tuple[float, float]]:
        """Extract and return person centroids from detections"""
        centroids = []
        h, w = self.frame_height, self.frame_width
        
        for person in people:
            x1, y1, x2, y2 = person.get('bbox', [0, 0, 0, 0])
            
            # Apply analysis zone if specified
            if self.analysis_zone:
                zx1, zy1, zx2, zy2 = self.analysis_zone
                if not (x2 > zx1 and x1 < zx2 and y2 > zy1 and y1 < zy2):
                    continue
            
            cx = (x1 + x2) / 2 / w  # Normalize 0-1
            cy = (y1 + y2) / 2 / h
            centroids.append((cx, cy))
            
        return centroids
    
    def calculate_density(self, centroids: List[Tuple[float, float]]) -> float:
        """Calculate crowd density (people per normalized area)"""
        if not centroids:
            return 0.0
        
        # Convex hull area for actual occupied space
        if len(centroids) < 3:
            return len(centroids)
        
        points = np.array(centroids, dtype=np.float32) * [self.frame_width, self.frame_height]
        hull = cv2.convexHull(points)
        hull_area = cv2.contourArea(hull)
        density = len(centroids) / (hull_area / (self.frame_width * self.frame_height))
        
        return density
    
    def calculate_motion_vectors(self, prev_centroids: List[Tuple[float, float]], 
                               curr_centroids: List[Tuple[float, float]]) -> List[float]:
        """Calculate speed and direction divergence"""
        speeds = []
        directions = []
        
        if not prev_centroids or not curr_centroids:
            return speeds
        
        # Simple nearest neighbor matching
        for prev in prev_centroids:
            distances = [((prev[0]-c[0])**2 + (prev[1]-c[1])**2)**0.5 for c in curr_centroids]
            if distances:
                nearest_idx = np.argmin(distances)
                dx = curr_centroids[nearest_idx][0] - prev[0]
                dy = curr_centroids[nearest_idx][1] - prev[1]
                
                speed = (dx**2 + dy**2)**0.5 * 60  # Scale to px/sec
                direction = np.degrees(np.arctan2(dy, dx))
                
                speeds.append(speed)
                directions.append(direction % 360)
        
        return speeds
    
    def detect_panic_motion(self, speeds: List[float], directions: List[float]) -> float:
        """Calculate panic score from motion patterns"""
        if not speeds:
            return 0.0
        
        mean_speed = np.mean(speeds)
        speed_variance = np.var(speeds)
        direction_std = np.std(directions) if directions else 0
        
        # Panic indicators:
        # 1. High average speed
        # 2. High speed variance (some running, some frozen)
        # 3. High direction variance (everyone running different ways)
        panic_score = 0.0
        
        if mean_speed > self.SPEED_THRESHOLD:
            panic_score += 0.3
        if speed_variance > 1.0:
            panic_score += 0.25
        if direction_std > self.DIRECTION_VARIANCE:
            panic_score += 0.3
        if len(speeds) > 10 and np.percentile(speeds, 75) > 3.0:
            panic_score += 0.15  # 75% moving fast
        
        return min(1.0, panic_score)
    
    def predict_stampede(self, density: float, panic_score: float, 
                        speed_trend: float) -> Optional[int]:
        """Predict time until stampede (15-60 seconds)"""
        if density < self.DENSITY_CRITICAL or panic_score < 0.6:
            return None
        
        # Simple linear prediction based on current trends
        base_time = 60
        density_factor = max(0, (density - self.DENSITY_CRITICAL) * 10)
        panic_factor = panic_score * 20
        speed_factor = speed_trend * 5
        
        predicted_seconds = max(15, base_time - density_factor - panic_factor - speed_factor)
        return int(predicted_seconds)
    
    def analyze(self, people: List[Dict], frame: np.ndarray = None) -> Dict:
        """Main analysis function"""
        self.frame_count += 1
        
        # Get current centroids
        curr_centroids = self.update_centroids(people)
        self.centroids_history.append(curr_centroids)
        
        # Calculate density
        density = self.calculate_density(curr_centroids)
        
        # Motion analysis (need previous frame)
        speeds = []
        if len(self.centroids_history) > 1:
            prev_centroids = list(self.centroids_history)[-2]
            speeds = self.calculate_motion_vectors(prev_centroids, curr_centroids)
            self.speeds_history.append(speeds)
        
        # Panic detection
        panic_score = self.detect_panic_motion(speeds, [])
        self.directions_history.append(panic_score)
        
        # Speed trend (last 10 frames)
        recent_speeds = [s for frame_speeds in list(self.speeds_history)[-10:] for s in frame_speeds]
        speed_trend = np.mean(np.diff(recent_speeds[-5:])) if len(recent_speeds) > 5 else 0
        
        # Generate alerts
        alerts = []
        
        # Density alerts
        if density > self.DENSITY_STAMPEDE:
            alerts.append(CrowdAlert("STAMPEDE", "CRITICAL", 0.95, time.time(), 
                                   density=density, panic_score=panic_score))
        elif density > self.DENSITY_CRITICAL:
            alerts.append(CrowdAlert("CRITICAL_DENSITY", "HIGH", 0.90, time.time(), 
                                   density=density, panic_score=panic_score))
        elif density > self.DENSITY_WARNING:
            alerts.append(CrowdAlert("HIGH_DENSITY", "MEDIUM", 0.75, time.time(), 
                                   density=density, panic_score=panic_score))
        
        # Panic motion
        if panic_score > self.PANIC_SCORE_THRESHOLD:
            countdown = self.predict_stampede(density, panic_score, speed_trend)
            alerts.append(CrowdAlert("PANIC_MOTION", "HIGH", panic_score, time.time(), 
                                   countdown, density, panic_score))
        
        # Draw visualization if frame provided
        if frame is not None:
            self._draw_analysis(frame, density, panic_score, alerts)
        
        return {
            "density": round(density, 2),
            "panic_score": round(panic_score, 2),
            "speed_trend": round(speed_trend, 2),
            "person_count": len(people),
            "alerts": alerts,
            "safe": len(alerts) == 0
        }
    
    def _draw_analysis(self, frame: np.ndarray, density: float, panic_score: float, 
                      alerts: List[CrowdAlert]):
        """Draw crowd analysis overlay"""
        h, w = frame.shape[:2]
        
        # Background status
        color = (0, 255, 0) if len(alerts) == 0 else (0, 0, 255)
        status = f"Density: {density:.1f} | Panic: {panic_score:.2f}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Critical alerts
        for i, alert in enumerate(alerts):
            y = 60 + i * 35
            alert_text = f"{alert.alert_type}: {alert.severity}"
            if alert.countdown:
                alert_text += f" ({alert.countdown}s)"
            cv2.putText(frame, alert_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 0, 255), 2)
        
        # Density heatmap
        overlay = frame.copy()
        heat_color = (0, 255, 255) if density > 2.0 else (0, 255, 0)
        cv2.rectangle(overlay, (w-120, 10), (w-10, 60), heat_color, -1)
        cv2.putText(overlay, f"{density:.1f}", (w-110, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (0,0,0), 2)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
