from collections import deque
from dataclasses import dataclass
import time
import numpy as np
import cv2
from typing import List, Tuple, Deque
from .config import Config

@dataclass
class TrackingWarning:
    message: str
    confidence: float
    timestamp: float

class MovementTracker:
    def __init__(self, config: Config):
        self.config = config
        self.calibrated = False
        self.calibration_data = []
        self.baseline = None
        
        # Strike tracking
        self.eye_deviations = []  # List of (start_time, end_time) tuples
        self.head_deviations = []
        self.eye_strikes = 0
        self.head_strikes = 0
        self.last_eye_warning = 0
        self.last_head_warning = 0
        self.last_strike_reset = time.time()
        
        # Current deviation tracking
        self.current_eye_deviation = None
        self.current_head_deviation = None
        
        # Warning display tracking
        self.active_warnings = []
        
        # 3D model points for head pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        # Camera matrix initialization
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4,1))

    def calibrate(self, landmarks) -> bool:
        """Collect calibration data and establish baseline"""
        if len(self.calibration_data) < self.config.CALIBRATION_FRAMES:
            self.calibration_data.append(landmarks)
            return False
        
        if not self.calibrated:
            self.baseline = self._calculate_baseline()
            self.calibrated = True
        return True

    def _calculate_baseline(self) -> dict:
        """Calculate baseline positions from calibration data"""
        if not self.calibration_data:
            return {}

        # Calculate average positions for key landmarks
        nose_positions = []
        left_eye_positions = []
        right_eye_positions = []

        for landmarks in self.calibration_data:
            nose_positions.append([
                landmarks.landmark[1].x,
                landmarks.landmark[1].y
            ])
            left_eye_positions.append([
                landmarks.landmark[133].x,
                landmarks.landmark[133].y
            ])
            right_eye_positions.append([
                landmarks.landmark[362].x,
                landmarks.landmark[362].y
            ])

        return {
            'nose': np.mean(nose_positions, axis=0),
            'left_eye': np.mean(left_eye_positions, axis=0),
            'right_eye': np.mean(right_eye_positions, axis=0)
        }

    def _check_strikes(self, deviations: list, current_time: float) -> bool:
        """Check if recent deviations constitute a strike"""
        # Reset strikes periodically
        if current_time - self.last_strike_reset > self.config.STRIKE_RESET_TIME:
            self.eye_strikes = max(0, self.eye_strikes - 1)  # Gradually reduce strikes
            self.head_strikes = max(0, self.head_strikes - 1)
            self.last_strike_reset = current_time

        # Filter recent deviations
        recent_deviations = [
            (start, end) for start, end in deviations 
            if current_time - end < self.config.DEVIATION_TIMEOUT
        ]
        
        # Count only deviations that lasted long enough
        significant_deviations = [
            (start, end) for start, end in recent_deviations
            if end - start >= self.config.MIN_DEVIATION_TIME
        ]
        
        return len(significant_deviations) >= self.config.DEVIATIONS_FOR_STRIKE

    def detect_gaze(self, frame, face_landmarks) -> List[TrackingWarning]:
        warnings = []
        current_time = time.time()
        h, w, _ = frame.shape

        def get_eye_ratio(eye_points):
            """Calculate eye position ratio"""
            eye_region = np.array([[p.x * w, p.y * h] for p in eye_points])
            center = np.mean(eye_region, axis=0)
            left = np.min(eye_region[:, 0])
            right = np.max(eye_region[:, 0])
            top = np.min(eye_region[:, 1])
            bottom = np.max(eye_region[:, 1])
            
            x_ratio = (left - center[0])/(center[0] - right)
            y_ratio = (center[1] - top)/(bottom - center[1])
            return x_ratio, y_ratio

        # Get eye landmarks
        left_eye = [face_landmarks.landmark[i] for i in range(362, 382)]
        right_eye = [face_landmarks.landmark[i] for i in range(133, 153)]
        
        deviation_detected = False
        
        for eye, eye_name in [(left_eye, "Left Eye"), (right_eye, "Right Eye")]:
            x_ratio, y_ratio = get_eye_ratio(eye)
            
            if x_ratio > self.config.EYE_RATIO_RIGHT:
                deviation_detected = True
                warnings.append(TrackingWarning(
                    f"{eye_name}: Looking Right",
                    min(x_ratio / self.config.EYE_RATIO_RIGHT, 1.0),
                    current_time
                ))
            elif x_ratio < self.config.EYE_RATIO_LEFT:
                deviation_detected = True
                warnings.append(TrackingWarning(
                    f"{eye_name}: Looking Left",
                    min(self.config.EYE_RATIO_LEFT / x_ratio, 1.0),
                    current_time
                ))
            elif y_ratio < self.config.EYE_RATIO_UP:
                deviation_detected = True
                warnings.append(TrackingWarning(
                    f"{eye_name}: Looking Up",
                    min(self.config.EYE_RATIO_UP / y_ratio, 1.0),
                    current_time
                ))

        # Track continuous deviations
        if deviation_detected:
            if self.current_eye_deviation is None:
                self.current_eye_deviation = current_time
        elif self.current_eye_deviation is not None:
            # Deviation ended, record it if it was long enough
            if current_time - self.current_eye_deviation >= self.config.MIN_DEVIATION_TIME:
                self.eye_deviations.append((self.current_eye_deviation, current_time))
            self.current_eye_deviation = None

        # Check for strikes
        if self._check_strikes(self.eye_deviations, current_time):
            self.eye_strikes += 1
            self.eye_deviations.clear()
            
            if self.eye_strikes >= self.config.STRIKES_FOR_WARNING:
                if current_time - self.last_eye_warning > self.config.WARNING_DISPLAY_TIME:
                    strike_warning = TrackingWarning(
                        f"⚠️ Please maintain eye contact! ({self.eye_strikes} strikes)",
                        1.0,
                        current_time + self.config.WARNING_DISPLAY_TIME
                    )
                    warnings.append(strike_warning)
                    self.active_warnings.append((strike_warning, current_time + self.config.WARNING_DISPLAY_TIME))
                    self.last_eye_warning = current_time

        return warnings

    def _initialize_camera_matrix(self, frame_size):
        """Initialize camera matrix based on frame size"""
        focal_length = frame_size[1]
        center = (frame_size[1]/2, frame_size[0]/2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

    def detect_head_movement(self, frame, face_landmarks) -> List[TrackingWarning]:
        warnings = []
        current_time = time.time()

        if self.camera_matrix is None:
            self._initialize_camera_matrix(frame.shape)

        # Get key facial landmarks for head pose
        image_points = np.array([
            [face_landmarks.landmark[1].x * frame.shape[1],   # Nose tip
             face_landmarks.landmark[1].y * frame.shape[0]],
            [face_landmarks.landmark[152].x * frame.shape[1], # Chin
             face_landmarks.landmark[152].y * frame.shape[0]],
            [face_landmarks.landmark[226].x * frame.shape[1], # Left eye corner
             face_landmarks.landmark[226].y * frame.shape[0]],
            [face_landmarks.landmark[446].x * frame.shape[1], # Right eye corner
             face_landmarks.landmark[446].y * frame.shape[0]],
            [face_landmarks.landmark[57].x * frame.shape[1],  # Left mouth corner
             face_landmarks.landmark[57].y * frame.shape[0]],
            [face_landmarks.landmark[287].x * frame.shape[1], # Right mouth corner
             face_landmarks.landmark[287].y * frame.shape[0]]
        ], dtype="double")

        # Calculate head pose
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, 
            image_points, 
            self.camera_matrix, 
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            # Convert rotation vector to Euler angles
            rotation_mat, _ = cv2.Rodrigues(rotation_vector)
            pose_mat = cv2.hconcat((rotation_mat, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            
            pitch, yaw, roll = [float(angle) for angle in euler_angles]

            deviation_detected = False
            
            # Check head angles against thresholds
            if abs(pitch) > self.config.HEAD_UP_THRESHOLD:
                deviation_detected = True
                direction = "up" if pitch > 0 else "down"
                warnings.append(TrackingWarning(
                    f"Head tilted {direction}",
                    min(abs(pitch) / 90.0, 1.0),
                    current_time
                ))
            
            if abs(yaw) > self.config.HEAD_LEFT_THRESHOLD:
                deviation_detected = True
                direction = "left" if yaw < 0 else "right"
                warnings.append(TrackingWarning(
                    f"Head turned {direction}",
                    min(abs(yaw) / 90.0, 1.0),
                    current_time
                ))

            # Track continuous head movements
            if deviation_detected:
                if self.current_head_deviation is None:
                    self.current_head_deviation = current_time
            elif self.current_head_deviation is not None:
                if current_time - self.current_head_deviation >= self.config.MIN_DEVIATION_TIME:
                    self.head_deviations.append((self.current_head_deviation, current_time))
                self.current_head_deviation = None

            # Check for strikes
            if self._check_strikes(self.head_deviations, current_time):
                self.head_strikes += 1
                self.head_deviations.clear()
                
                if self.head_strikes >= self.config.STRIKES_FOR_WARNING:
                    if current_time - self.last_head_warning > self.config.WARNING_DISPLAY_TIME:
                        strike_warning = TrackingWarning(
                            f"⚠️ Please face forward! ({self.head_strikes} strikes)",
                            1.0,
                            current_time + self.config.WARNING_DISPLAY_TIME
                        )
                        warnings.append(strike_warning)
                        self.active_warnings.append((strike_warning, current_time + self.config.WARNING_DISPLAY_TIME))
                        self.last_head_warning = current_time

        return warnings