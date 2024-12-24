import os
from dataclasses import dataclass

@dataclass
class Config:
    CAMERA_INDEX: int = 0
    DESIRED_FPS: int = 10
    
    # Eye tracking thresholds (adjusted for better sensitivity)
    EYE_HORIZONTAL_THRESHOLD: float = 0.20    # More sensitive horizontal detection
    EYE_VERTICAL_THRESHOLD: float = 0.15      # More sensitive vertical detection
    BLINK_RATIO_THRESHOLD: float = 4.0        # Adjusted blink detection
    RAPID_EYE_MOVEMENT_THRESHOLD: int = 3      # Quick eye movements in 2 seconds
    EYE_ASPECT_RATIO_THRESHOLD: float = 0.2    # For blink detection
    
    # Head movement thresholds (in degrees)
    HEAD_ROTATION_THRESHOLD: float = 20.0    # Side to side movement
    HEAD_TILT_THRESHOLD: float = 15.0       # Up/down movement
    HEAD_SIDE_THRESHOLD: float = 25.0       # Left/right lean
    
    # Suspicious behavior detection
    SUSTAINED_LOOK_TIME: float = 1.5        # Sustained looking away time
    PATTERN_DETECTION_WINDOW: float = 5.0    # Time window for pattern detection
    RAPID_HEAD_MOVEMENT_THRESHOLD: int = 3   # Quick head movements in 3 seconds
    
    # Warning system
    WARNING_DISPLAY_TIME: float = 3.0
    DEVIATION_TIMEOUT: float = 4.0
    DEVIATIONS_FOR_STRIKE: int = 3
    STRIKES_FOR_WARNING: int = 3
    STRIKE_RESET_TIME: float = 60.0         # Reset strikes after 1 minute
    MIN_DEVIATION_TIME: float = 0.8         # Minimum time for suspicious movement
    
    # Visualization
    LANDMARK_COLOR: tuple = (0, 255, 0)     # Green color for landmarks
    WARNING_COLOR: tuple = (0, 0, 255)      # Red color for warnings
    ANNOTATION_THICKNESS: int = 2
    
    # Gaze tracking
    GAZE_RIGHT_THRESHOLD: float = 0.35        # Right gaze threshold
    GAZE_LEFT_THRESHOLD: float = -0.35        # Left gaze threshold
    GAZE_UP_THRESHOLD: float = -0.30          # Upward gaze threshold
    GAZE_DOWN_THRESHOLD: float = 0.30         # Downward gaze threshold