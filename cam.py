import cv2
import mediapipe as mp
import numpy as np
import json
import os

class EightDirectionEyeTracker:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Window setup
        self.window_width = 1280
        self.window_height = 720
        cv2.namedWindow('8-Direction Eye Tracker', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('8-Direction Eye Tracker', self.window_width, self.window_height)

        # Cursor parameters
        self.cursor_x = self.window_width // 2
        self.cursor_y = self.window_height // 2
        self.movement_unit = 10  # Base movement unit in pixels
        self.vertical_sensitivity = 1.0
        self.horizontal_sensitivity = 1.0
        self.smoothing = 0.2  # Smoothing factor

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Calibration and reference points
        self.center_left_eye = None
        self.center_right_eye = None
        self.calibrated = False

        # Smooth movement tracking
        self.moving_avg_x = self.window_width // 2
        self.moving_avg_y = self.window_height // 2

        # Direction threshold (minimum movement to detect direction)
        self.direction_threshold = 0.01

        # Eye landmarks
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]

        # Load configuration if it exists
        self.load_configuration()

    def load_configuration(self):
        config_file = "config.json"
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
                self.movement_unit = config.get("movement_unit", 10)
                self.vertical_sensitivity = config.get("vertical_sensitivity", 1.0)
                self.horizontal_sensitivity = config.get("horizontal_sensitivity", 1.0)
                print("Loaded configuration from file.")

    def save_configuration(self):
        config = {
            "movement_unit": self.movement_unit,
            "vertical_sensitivity": self.vertical_sensitivity,
            "horizontal_sensitivity": self.horizontal_sensitivity
        }
        with open("config.json", "w") as f:
            json.dump(config, f, indent=4)
        print("Configuration saved.")

    def get_eye_centers(self, frame, landmarks):
        frame_h, frame_w = frame.shape[:2]

        left_iris = np.array([(landmarks.landmark[i].x * frame_w,
                               landmarks.landmark[i].y * frame_h)
                              for i in self.LEFT_IRIS])
        right_iris = np.array([(landmarks.landmark[i].x * frame_w,
                                landmarks.landmark[i].y * frame_h)
                               for i in self.RIGHT_IRIS])

        left_center = np.mean(left_iris, axis=0)
        right_center = np.mean(right_iris, axis=0)

        return left_center, right_center

    def calibrate(self, left_center, right_center):
        self.center_left_eye = left_center
        self.center_right_eye = right_center
        self.cursor_x = self.window_width // 2
        self.cursor_y = self.window_height // 2
        self.calibrated = True
        print(f"Calibrated - Movement unit: {self.movement_unit} pixels")

    def get_direction(self, dx, dy):
        # Determine the direction based on movement
        if abs(dx) < self.direction_threshold and abs(dy) < self.direction_threshold:
            return "Center"

        if abs(dx) > abs(dy) * 2:
            return "Left" if dx > 0 else "Right"
        elif abs(dy) > abs(dx) * 2:
            return "Down" if dy > 0 else "Up"
        else:
            if dx > 0 and dy < 0:
                return "Top-Left"
            elif dx > 0 and dy > 0:
                return "Bottom-Left"
            elif dx < 0 and dy < 0:
                return "Top-Right"
            else:
                return "Bottom-Right"

    def calculate_movement(self, left_center, right_center):
        if not self.calibrated:
            return 0, 0

        # Calculate deviations
        left_dev_x = left_center[0] - self.center_left_eye[0]
        right_dev_x = right_center[0] - self.center_right_eye[0]
        left_dev_y = left_center[1] - self.center_left_eye[1]
        right_dev_y = right_center[1] - self.center_right_eye[1]

        # Average deviation scaled by movement unit and sensitivities
        dx = ((left_dev_x + right_dev_x) / 2) * self.movement_unit * self.horizontal_sensitivity
        dy = ((left_dev_y + right_dev_y) / 2) * self.movement_unit * self.vertical_sensitivity

        return dx, dy

    def smooth_movement(self, x, y):
        self.moving_avg_x = (self.moving_avg_x * (1 - self.smoothing) +
                             x * self.smoothing)
        self.moving_avg_y = (self.moving_avg_y * (1 - self.smoothing) +
                             y * self.smoothing)
        return self.moving_avg_x, self.moving_avg_y

    def draw_interface(self, frame, direction="Center"):
        # Draw cursor
        cursor_size = 20
        cv2.line(frame,
                 (int(self.cursor_x - cursor_size), int(self.cursor_y)),
                 (int(self.cursor_x + cursor_size), int(self.cursor_y)),
                 (0, 255, 0), 2)
        cv2.line(frame,
                 (int(self.cursor_x), int(self.cursor_y - cursor_size)),
                 (int(self.cursor_x), int(self.cursor_y + cursor_size)),
                 (0, 255, 0), 2)

        # Draw center reference
        center_x = self.window_width // 2
        center_y = self.window_height // 2
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

        # Display information
        cv2.putText(frame, f"Movement Unit: {self.movement_unit}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Direction: {direction}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Vertical Sensitivity: {self.vertical_sensitivity:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Horizontal Sensitivity: {self.horizontal_sensitivity:.2f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "i/d: increase/decrease movement", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "w/s: increase/decrease vertical sensitivity", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "a/f: increase/decrease horizontal sensitivity", (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "s: save configuration", (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def run(self):
        print("Starting 8-direction eye tracker...")
        print("Look at the center blue dot and press 'c' to calibrate")
        print("Press 'i' to increase movement unit")
        print("Press 'd' to decrease movement unit")
        print("Press 'w' to increase vertical sensitivity")
        print("Press 's' to save configuration")
        print("Press 'a' to increase horizontal sensitivity")
        print("Press 'f' to decrease horizontal sensitivity")
        print("Press 'q' to quit")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)  # Mirror flip
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)

                direction = "Center"
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    left_center, right_center = self.get_eye_centers(frame, landmarks)

                    if self.calibrated:
                        # Calculate and apply movement
                        dx, dy = self.calculate_movement(left_center, right_center)
                        smooth_x, smooth_y = self.smooth_movement(dx, dy)

                        # Update cursor position
                        new_x = self.window_width // 2 + smooth_x
                        new_y = self.window_height // 2 + smooth_y

                        self.cursor_x = max(0, min(self.window_width, new_x))
                        self.cursor_y = max(0, min(self.window_height, new_y))

                        # Get and display direction
                        direction = self.get_direction(dx / self.movement_unit,
                                                       dy / self.movement_unit)

                    # Draw eye centers
                    cv2.circle(frame, (int(left_center[0]), int(left_center[1])),
                               2, (255, 0, 0), -1)
                    cv2.circle(frame, (int(right_center[0]), int(right_center[1])),
                               2, (255, 0, 0), -1)

                self.draw_interface(frame, direction)
                cv2.imshow('8-Direction Eye Tracker', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    if results.multi_face_landmarks:
                        self.calibrate(left_center, right_center)
                elif key == ord('i'):
                    self.movement_unit = min(50, self.movement_unit + 1)
                    print(f"Movement unit increased to: {self.movement_unit}")
                elif key == ord('d'):
                    self.movement_unit = max(1, self.movement_unit - 1)
                    print(f"Movement unit decreased to: {self.movement_unit}")
                elif key == ord('w'):
                    self.vertical_sensitivity = min(5.0, self.vertical_sensitivity + 0.1)
                    print(f"Vertical sensitivity increased to: {self.vertical_sensitivity:.2f}")
                elif key == ord('s'):
                    self.save_configuration()  # Save configuration when 's' is pressed
                elif key == ord('a'):
                    self.horizontal_sensitivity = min(5.0, self.horizontal_sensitivity + 0.1)
                    print(f"Horizontal sensitivity increased to: {self.horizontal_sensitivity:.2f}")
                elif key == ord('f'):
                    self.horizontal_sensitivity = max(0.1, self.horizontal_sensitivity - 0.1)
                    print(f"Horizontal sensitivity decreased to: {self.horizontal_sensitivity:.2f}")

        except Exception as e:
            print(f"Error occurred: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()



