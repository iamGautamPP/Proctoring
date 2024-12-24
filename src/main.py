import cv2
import time
import mediapipe as mp
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from .config import Config
from .eye_tracker import EyeAnalyzer
from .head_tracker import HeadTracker

class InterviewMonitor:
    def __init__(self):
        self.config = Config()
        self.face_mesh = self._initialize_face_mesh()
        self.eye_analyzer = EyeAnalyzer(self.config)
        self.head_tracker = HeadTracker(self.config)
        self.setup_logging()

    def _initialize_face_mesh(self):
        mp_face_mesh = mp.solutions.face_mesh
        return mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def setup_logging(self):
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)
        
        # Create a new log file for each session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/interview_session_{timestamp}.log"
        
        # Configure file handler only (remove StreamHandler)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file)
            ]
        )

    def log_suspicious_behavior(self, warnings):
        for warning, confidence in warnings:
            logging.warning(f"{warning} (Confidence: {confidence:.2f})")

    def run(self):
        try:
            cap = cv2.VideoCapture(self.config.CAMERA_INDEX)
            if not cap.isOpened():
                raise RuntimeError("Failed to open camera feed")

            frame_delay = 1 / self.config.DESIRED_FPS
            session_start = time.time()

            while True:
                frame_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]

                    try:
                        # Analyze eye movements
                        eye_warnings = self.eye_analyzer.detect_suspicious_patterns(
                            frame, face_landmarks
                        )
                        
                        # Analyze head movements
                        head_warnings = self.head_tracker.detect_suspicious_head_movements(
                            frame, face_landmarks
                        )

                        # Combine and log warnings
                        all_warnings = eye_warnings + head_warnings
                        if all_warnings:
                            self.log_suspicious_behavior(all_warnings)

                        # Display warnings on frame
                        y_offset = 30
                        for warning, confidence in all_warnings:
                            cv2.putText(
                                frame,
                                f"{warning} ({confidence:.2f})",
                                (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255) if confidence > 0.8 else (0, 255, 255),
                                2
                            )
                            y_offset += 30

                    except Exception as e:
                        logging.error(f"Error processing frame: {str(e)}")
                        continue

                cv2.imshow("Interview Monitor", frame)

                # Control frame rate
                processing_time = time.time() - frame_start
                if processing_time < frame_delay:
                    time.sleep(frame_delay - processing_time)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            logging.error(f"Error during monitoring: {str(e)}")
        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor = InterviewMonitor()
    monitor.run()