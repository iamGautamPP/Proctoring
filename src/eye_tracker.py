import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import time

@dataclass
class EyeRegion:
    left: float
    top: float
    right: float
    bottom: float
    center_x: float
    center_y: float
    landmarks: List[tuple]

class EyeAnalyzer:
    def __init__(self, config):
        self.config = config
        # Landmark indices for eyes
        self.LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Indices for iris detection
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        self.last_blink_time = 0
        self.blink_count = 0
        self.rapid_movements = []
        self.last_gaze_time = time.time()
        self.sustained_gaze = None

    def get_eye_region(self, landmarks, indices, frame_shape) -> EyeRegion:
        points = []
        for idx in indices:
            landmark = landmarks.landmark[idx]
            x, y = int(landmark.x * frame_shape[1]), int(landmark.y * frame_shape[0])
            points.append((x, y))
        
        points = np.array(points)
        left, top = points.min(axis=0)
        right, bottom = points.max(axis=0)
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        
        return EyeRegion(left, top, right, bottom, center_x, center_y, points)

    def calculate_ear(self, eye_region: EyeRegion) -> float:
        """Calculate Eye Aspect Ratio"""
        # Vertical eye landmarks
        v1 = np.linalg.norm(eye_region.landmarks[1] - eye_region.landmarks[5])
        v2 = np.linalg.norm(eye_region.landmarks[2] - eye_region.landmarks[4])
        # Horizontal eye landmarks
        h = np.linalg.norm(eye_region.landmarks[0] - eye_region.landmarks[3])
        # EAR
        ear = (v1 + v2) / (2.0 * h) if h > 0 else 0
        return ear

    def get_iris_position(self, landmarks, iris_indices, eye_region: EyeRegion) -> tuple:
        """Calculate iris position relative to eye region"""
        iris_points = []
        for idx in iris_indices:
            landmark = landmarks.landmark[idx]
            x = landmark.x * (eye_region.right - eye_region.left) + eye_region.left
            y = landmark.y * (eye_region.bottom - eye_region.top) + eye_region.top
            iris_points.append((x, y))
        
        iris_center = np.mean(iris_points, axis=0)
        eye_width = eye_region.right - eye_region.left
        relative_x = (iris_center[0] - eye_region.left) / eye_width if eye_width > 0 else 0.5
        return relative_x, iris_center

    def detect_suspicious_patterns(self, frame, landmarks) -> List[Tuple[str, float]]:
        warnings = []
        current_time = time.time()
        h, w = frame.shape[:2]

        # Get eye regions
        left_eye = self.get_eye_region(landmarks, self.LEFT_EYE_INDICES, frame.shape)
        right_eye = self.get_eye_region(landmarks, self.RIGHT_EYE_INDICES, frame.shape)

        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Get iris positions
        left_iris_x, left_iris_center = self.get_iris_position(landmarks, self.LEFT_IRIS, left_eye)
        right_iris_x, right_iris_center = self.get_iris_position(landmarks, self.RIGHT_IRIS, right_eye)
        avg_iris_x = (left_iris_x + right_iris_x) / 2

        # Draw eye regions and iris centers
        cv2.polylines(frame, [left_eye.landmarks], True, self.config.LANDMARK_COLOR, 1)
        cv2.polylines(frame, [right_eye.landmarks], True, self.config.LANDMARK_COLOR, 1)
        cv2.circle(frame, (int(left_iris_center[0]), int(left_iris_center[1])), 2, (0, 255, 255), -1)
        cv2.circle(frame, (int(right_iris_center[0]), int(right_iris_center[1])), 2, (0, 255, 255), -1)

        # Detect blinks
        if avg_ear < self.config.EYE_ASPECT_RATIO_THRESHOLD:
            if current_time - self.last_blink_time > 0.5:
                self.blink_count += 1
                self.last_blink_time = current_time
                if self.blink_count >= 5:
                    warnings.append(("Excessive blinking detected", 0.9))
                    self.blink_count = 0

        # Detect gaze direction
        if avg_iris_x > 0.65:  # Looking right
            direction = "right"
            confidence = min((avg_iris_x - 0.5) * 2, 1.0)
            warnings.append((f"Looking {direction}", confidence))
        elif avg_iris_x < 0.35:  # Looking left
            direction = "left"
            confidence = min((0.5 - avg_iris_x) * 2, 1.0)
            warnings.append((f"Looking {direction}", confidence))

        # Track sustained gaze
        if warnings and self.sustained_gaze is None:
            self.sustained_gaze = (current_time, warnings[0][0])
        elif warnings and self.sustained_gaze is not None:
            if current_time - self.sustained_gaze[0] > self.config.SUSTAINED_LOOK_TIME:
                warnings.append(("Sustained irregular gaze detected", 0.95))
        else:
            self.sustained_gaze = None

        # Track rapid movements
        if warnings:
            self.rapid_movements.append(current_time)
            self.rapid_movements = [t for t in self.rapid_movements 
                                  if current_time - t < self.config.PATTERN_DETECTION_WINDOW]
            
            if len(self.rapid_movements) >= self.config.RAPID_EYE_MOVEMENT_THRESHOLD:
                warnings.append(("Suspicious rapid eye movements", 0.95))
                self.rapid_movements.clear()

        return warnings

    def draw_eye_annotations(self, frame, eye_region: EyeRegion, is_left: bool):
        """Draw eye region and gaze direction"""
        # Draw eye region
        cv2.polylines(frame, [eye_region.landmarks], True, self.config.LANDMARK_COLOR, 1)
        
        # Draw eye center
        cv2.circle(frame, (int(eye_region.center_x), int(eye_region.center_y)), 
                   2, (0, 255, 255), -1)
