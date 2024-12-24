import cv2
import numpy as np
from typing import List, Tuple
import time

class HeadTracker:
    def __init__(self, config):
        self.config = config
        
        # 3D model points for head pose estimation (in millimeters)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        # Camera matrix initialization
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4,1), dtype=np.float64)
        
        # Movement tracking
        self.head_movements = []
        self.last_position = None
        self.movement_start_time = None
        self.suspicious_movements = []

    def _initialize_camera_matrix(self, frame_shape):
        if self.camera_matrix is None:
            focal_length = frame_shape[1]
            center = (frame_shape[1]/2, frame_shape[0]/2)
            self.camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

    def draw_head_pose_annotations(self, frame, rotation_vector, translation_vector):
        """Draw 3D axes showing head orientation"""
        try:
            points = np.array([
                (0, 0, 0),      # Origin
                (200, 0, 0),    # X axis
                (0, 200, 0),    # Y axis
                (0, 0, 200)     # Z axis
            ], dtype=np.float64).reshape(-1, 3)

            points_2d, _ = cv2.projectPoints(
                points, 
                rotation_vector, 
                translation_vector, 
                self.camera_matrix, 
                self.dist_coeffs
            )
            
            if points_2d is not None:
                points_2d = points_2d.reshape(-1, 2).astype(np.int32)
                origin = tuple(points_2d[0])
                cv2.line(frame, origin, tuple(points_2d[1]), (0, 0, 255), 3)  # X axis - Red
                cv2.line(frame, origin, tuple(points_2d[2]), (0, 255, 0), 3)  # Y axis - Green
                cv2.line(frame, origin, tuple(points_2d[3]), (255, 0, 0), 3)  # Z axis - Blue
        except Exception as e:
            print(f"Error in draw_head_pose_annotations: {str(e)}")

    def detect_suspicious_head_movements(self, frame, face_landmarks) -> List[Tuple[str, float]]:
        warnings = []
        current_time = time.time()
        
        self._initialize_camera_matrix(frame.shape)
        
        try:
            # Get key facial landmarks and ensure they're float64
            image_points = np.array([
                [face_landmarks.landmark[1].x * frame.shape[1],    # Nose tip
                 face_landmarks.landmark[1].y * frame.shape[0]],
                [face_landmarks.landmark[152].x * frame.shape[1],  # Chin
                 face_landmarks.landmark[152].y * frame.shape[0]],
                [face_landmarks.landmark[226].x * frame.shape[1],  # Left eye corner
                 face_landmarks.landmark[226].y * frame.shape[0]],
                [face_landmarks.landmark[446].x * frame.shape[1],  # Right eye corner
                 face_landmarks.landmark[446].y * frame.shape[0]],
                [face_landmarks.landmark[57].x * frame.shape[1],   # Left mouth corner
                 face_landmarks.landmark[57].y * frame.shape[0]],
                [face_landmarks.landmark[287].x * frame.shape[1],  # Right mouth corner
                 face_landmarks.landmark[287].y * frame.shape[0]]
            ], dtype=np.float64)

            # Ensure we have valid landmarks
            if np.any(np.isnan(image_points)):
                return warnings

            # Calculate head pose
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points, 
                image_points, 
                self.camera_matrix, 
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                # Draw head pose visualization
                self.draw_head_pose_annotations(frame, rotation_vector, translation_vector)
                
                # Convert rotation vector to Euler angles
                rotation_mat, _ = cv2.Rodrigues(rotation_vector)
                pose_mat = cv2.hconcat((rotation_mat, translation_vector))
                _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
                
                pitch, yaw, roll = [float(angle) for angle in euler_angles]

                # Track current position
                current_position = (pitch, yaw, roll)
                
                # Rest of the movement detection code...
                if self.last_position is not None:
                    movement = np.array(current_position) - np.array(self.last_position)
                    movement_magnitude = np.linalg.norm(movement)
                    
                    if movement_magnitude > self.config.HEAD_ROTATION_THRESHOLD:
                        if self.movement_start_time is None:
                            self.movement_start_time = current_time
                        elif current_time - self.movement_start_time >= self.config.MIN_DEVIATION_TIME:
                            self.suspicious_movements.append(current_time)
                    else:
                        self.movement_start_time = None

                self.last_position = current_position
                
                # Check angles and generate warnings
                if abs(pitch) > self.config.HEAD_TILT_THRESHOLD:
                    direction = "up" if pitch > 0 else "down"
                    confidence = min(abs(pitch) / 90.0, 1.0)
                    warnings.append((f"Head tilted {direction}", confidence))

                if abs(yaw) > self.config.HEAD_ROTATION_THRESHOLD:
                    direction = "right" if yaw > 0 else "left"
                    confidence = min(abs(yaw) / 90.0, 1.0)
                    warnings.append((f"Head turned {direction}", confidence))

                if abs(roll) > self.config.HEAD_SIDE_THRESHOLD:
                    direction = "right" if roll > 0 else "left"
                    confidence = min(abs(roll) / 90.0, 1.0)
                    warnings.append((f"Head leaning {direction}", confidence))

        except Exception as e:
            print(f"Error in detect_suspicious_head_movements: {str(e)}")
            
        return warnings
