import cv2
import mediapipe as mp
import numpy as np
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename='logs/tracking.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def initialize_face_mesh():
    """Initialize MediaPipe Face Mesh"""
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def log_warning(warning_msg: str, confidence: float):
    """Log warnings with timestamp and confidence score"""
    if "strikes" in warning_msg:
        logging.warning(f"STRIKE ALERT - {warning_msg} (Confidence: {confidence:.2f})")
    else:
        logging.warning(f"{warning_msg} (Confidence: {confidence:.2f})")