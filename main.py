import tkinter as tk
from tkinter import messagebox
import subprocess

import cv2
import mediapipe as mp
import numpy as np
import json
import os


import cv2
import mediapipe as mp
import numpy as np
import time


import cv2
import mediapipe as mp
import numpy as np

class EyeTrackingGame:
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
        self.window_name = 'Eye Tracking Game'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)

        # Cursor parameters
        self.cursor_x = self.window_width // 2
        self.cursor_y = self.window_height // 2
        self.movement_unit = 10
        self.vertical_sensitivity = 1.0
        self.horizontal_sensitivity = 1.0
        self.smoothing = 0.2

        # Game objects
        self.objects = [
            {"pos": (100, 100), "color": (0, 0, 255), "name": "Red", "clicked": False},
            {"pos": (1180, 100), "color": (0, 255, 0), "name": "Green", "clicked": False},
            {"pos": (100, 620), "color": (255, 0, 0), "name": "Blue", "clicked": False},
            {"pos": (1180, 620), "color": (255, 255, 0), "name": "Yellow", "clicked": False}
        ]
        self.object_radius = 30
        self.score = 0

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

        # Eye landmarks
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]

        # Create blank canvas
        self.canvas = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)

    def get_eye_centers(self, landmarks):
        left_iris = np.array([(landmarks.landmark[i].x * self.window_width,
                              landmarks.landmark[i].y * self.window_height)
                             for i in self.LEFT_IRIS])
        right_iris = np.array([(landmarks.landmark[i].x * self.window_width,
                               landmarks.landmark[i].y * self.window_height)
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

    def calculate_movement(self, left_center, right_center):
        if not self.calibrated:
            return 0, 0

        left_dev_x = left_center[0] - self.center_left_eye[0]
        right_dev_x = right_center[0] - self.center_right_eye[0]
        left_dev_y = left_center[1] - self.center_left_eye[1]
        right_dev_y = right_center[1] - self.center_right_eye[1]

        dx = ((left_dev_x + right_dev_x) / 2) * self.movement_unit * self.horizontal_sensitivity
        dy = ((left_dev_y + right_dev_y) / 2) * self.movement_unit * self.vertical_sensitivity

        return dx, dy

    def smooth_movement(self, x, y):
        self.moving_avg_x = (self.moving_avg_x * (1 - self.smoothing) + x * self.smoothing)
        self.moving_avg_y = (self.moving_avg_y * (1 - self.smoothing) + y * self.smoothing)
        return self.moving_avg_x, self.moving_avg_y

    def check_object_collision(self):
        for obj in self.objects:
            if not obj["clicked"]:
                distance = np.sqrt((self.cursor_x - obj["pos"][0])**2 +
                                 (self.cursor_y - obj["pos"][1])**2)
                if distance < self.object_radius:
                    obj["clicked"] = True
                    self.score += 1
                    print(f"Clicked {obj['name']}! Score: {self.score}")
                    if self.score == len(self.objects):
                        print("Congratulations! You've clicked all objects!")
                        return True
        return False

    def draw_interface(self):
        # Clear canvas
        self.canvas.fill(0)

        # Draw objects
        for obj in self.objects:
            color = obj["color"] if not obj["clicked"] else (128, 128, 128)
            cv2.circle(self.canvas, obj["pos"], self.object_radius, color, -1)

        # Draw cursor
        cursor_size = 20
        cv2.line(self.canvas,
                 (int(self.cursor_x - cursor_size), int(self.cursor_y)),
                 (int(self.cursor_x + cursor_size), int(self.cursor_y)),
                 (0, 255, 0), 2)
        cv2.line(self.canvas,
                 (int(self.cursor_x), int(self.cursor_y - cursor_size)),
                 (int(self.cursor_x), int(self.cursor_y + cursor_size)),
                 (0, 255, 0), 2)

        # Draw score and instructions
        cv2.putText(self.canvas, f"Score: {self.score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if not self.calibrated:
            cv2.putText(self.canvas, "Look at center and press 'c' to calibrate",
                       (self.window_width//2 - 200, self.window_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return self.canvas

    def run(self):
        print("Starting Eye Tracking Game...")
        print("Look at the center and press 'c' to calibrate")
        print("Move your eyes to control the cursor")
        print("Click objects by looking at them")
        print("Press 'q' to quit")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    left_center, right_center = self.get_eye_centers(landmarks)

                    if self.calibrated:
                        dx, dy = self.calculate_movement(left_center, right_center)
                        smooth_x, smooth_y = self.smooth_movement(dx, dy)

                        new_x = self.window_width // 2 + smooth_x
                        new_y = self.window_height // 2 + smooth_y

                        self.cursor_x = max(0, min(self.window_width, new_x))
                        self.cursor_y = max(0, min(self.window_height, new_y))

                        if self.check_object_collision():
                            print("Game Complete!")

                # Draw game interface
                game_display = self.draw_interface()
                cv2.imshow(self.window_name, game_display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c') and results.multi_face_landmarks:
                    self.calibrate(left_center, right_center)

        except Exception as e:
            print(f"Error occurred: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

class EyeTrackingKeyboard:
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
        cv2.namedWindow('Eye-Tracking Keyboard', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Eye-Tracking Keyboard', self.window_width, self.window_height)

        # Keyboard layout and settings
        self.keys = [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/'],
            ['SHIFT', 'SPACE', 'BACKSPACE', 'ENTER', 'CLEAR']
        ]
        self.key_width = 100
        self.key_height = 80
        self.key_spacing = 10
        self.typed_text = ""
        self.holding_time = 0
        self.hold_threshold = 2.0  # seconds
        self.last_key = None
        self.last_key_time = 0

        # Cursor parameters
        self.cursor_x = self.window_width // 2
        self.cursor_y = self.window_height // 2
        self.movement_unit = 10
        self.vertical_sensitivity = 1.0
        self.horizontal_sensitivity = 1.0
        self.smoothing = 0.2

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

        # Eye landmarks
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

        # Blink detection
        self.blink_threshold = 0.2
        self.last_blink_time = time.time()
        self.blink_cooldown = 0.5  # seconds

    def calculate_ear(self, landmarks, eye_indices):
        points = []
        for i in eye_indices:
            point = [landmarks.landmark[i].x, landmarks.landmark[i].y]
            points.append(point)

        # Calculate EAR
        ear = (self.distance(points[1], points[5]) + self.distance(points[2], points[4])) / (
                    2 * self.distance(points[0], points[3]))
        return ear

    def distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def detect_blink(self, landmarks):
        left_ear = self.calculate_ear(landmarks, self.LEFT_EYE)
        right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2

        current_time = time.time()
        if avg_ear < self.blink_threshold and (current_time - self.last_blink_time) > self.blink_cooldown:
            self.last_blink_time = current_time
            return True
        return False

    def draw_keyboard(self, frame):
        # Calculate starting position to center the keyboard
        start_x = (self.window_width - (len(self.keys[0]) * (self.key_width + self.key_spacing))) // 2
        start_y = self.window_height - (len(self.keys) * (self.key_height + self.key_spacing)) - 50

        # Draw each key
        for row_idx, row in enumerate(self.keys):
            for col_idx, key in enumerate(row):
                x = start_x + col_idx * (self.key_width + self.key_spacing)
                y = start_y + row_idx * (self.key_height + self.key_spacing)

                # Special handling for bottom row
                if row_idx == len(self.keys) - 1:
                    if key in ['SHIFT', 'SPACE', 'BACKSPACE', 'ENTER', 'CLEAR']:
                        width = self.key_width * 2
                    else:
                        width = self.key_width
                else:
                    width = self.key_width

                # Check if cursor is over this key
                is_selected = (x < self.cursor_x < x + width and
                               y < self.cursor_y < y + self.key_height)

                # Draw key background
                color = (100, 200, 100) if is_selected else (50, 50, 50)
                cv2.rectangle(frame, (x, y), (x + width, y + self.key_height), color, -1)
                cv2.rectangle(frame, (x, y), (x + width, y + self.key_height), (200, 200, 200), 1)

                # Draw key text
                text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = x + (width - text_size[0]) // 2
                text_y = y + (self.key_height + text_size[1]) // 2
                cv2.putText(frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Update holding time for selected key
                if is_selected:
                    if self.last_key != key:
                        self.holding_time = time.time()
                        self.last_key = key
                    elif time.time() - self.holding_time >= self.hold_threshold:
                        self.process_key(key)
                        self.holding_time = time.time()

        # Draw typed text area
        cv2.rectangle(frame, (50, 50), (self.window_width - 50, 150), (30, 30, 30), -1)
        cv2.rectangle(frame, (50, 50), (self.window_width - 50, 150), (200, 200, 200), 1)
        cv2.putText(frame, self.typed_text[-80:], (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def process_key(self, key):
        if key == 'BACKSPACE':
            self.typed_text = self.typed_text[:-1]
        elif key == 'SPACE':
            self.typed_text += ' '
        elif key == 'ENTER':
            self.typed_text += '\n'
        elif key == 'CLEAR':
            self.typed_text = ''
        elif key == 'SHIFT':
            pass  # Implement shift functionality if needed
        else:
            self.typed_text += key

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
        print("Calibration complete - Look at keys and hold gaze or blink to select")

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

    def draw_interface(self, frame):
        # Draw cursor
        cursor_size = 10
        cv2.circle(frame, (int(self.cursor_x), int(self.cursor_y)), cursor_size, (0, 255, 0), 2)

        # Draw calibration instructions if not calibrated
        if not self.calibrated:
            cv2.putText(frame, "Look at the center and press 'c' to calibrate",
                        (self.window_width // 4, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.circle(frame, (self.window_width // 2, self.window_height // 2),
                       10, (255, 0, 0), -1)

        # Draw controls info
        controls = [
            "Controls:",
            "i/d: increase/decrease movement speed",
            "w/s: increase/decrease vertical sensitivity",
            "a/f: increase/decrease horizontal sensitivity",
            "q: quit",
            f"Movement Speed: {self.movement_unit}",
            f"V-Sensitivity: {self.vertical_sensitivity:.1f}",
            f"H-Sensitivity: {self.horizontal_sensitivity:.1f}"
        ]

        for i, text in enumerate(controls):
            cv2.putText(frame, text, (10, 200 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    def run(self):
        print("Starting Eye-Tracking Keyboard...")
        print("Look at the center blue dot and press 'c' to calibrate")
        print("Hold gaze for 3 seconds or blink to select a key")
        print("Press 'q' to quit")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)  # Mirror flip
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)

                # Create blank frame
                display = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    left_center, right_center = self.get_eye_centers(frame, landmarks)

                    # Handle blink detection
                    if self.detect_blink(landmarks) and self.calibrated:
                        # Find and process the key under cursor
                        self.process_current_key()

                    if self.calibrated:
                        # Calculate and apply movement
                        dx, dy = self.calculate_movement(left_center, right_center)
                        smooth_x, smooth_y = self.smooth_movement(dx, dy)

                        # Update cursor position
                        new_x = self.window_width // 2 + smooth_x
                        new_y = self.window_height // 2 + smooth_y

                        self.cursor_x = max(0, min(self.window_width, new_x))
                        self.cursor_y = max(0, min(self.window_height, new_y))

                self.draw_keyboard(display)
                self.draw_interface(display)

                cv2.imshow('Eye-Tracking Keyboard', display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c') and results.multi_face_landmarks:
                    self.calibrate(left_center, right_center)
                elif key == ord('i'):
                    self.movement_unit = min(50, self.movement_unit + 1)
                elif key == ord('d'):
                    self.movement_unit = max(1, self.movement_unit - 1)
                elif key == ord('w'):
                    self.vertical_sensitivity = min(5.0, self.vertical_sensitivity + 0.1)
                elif key == ord('s'):
                    self.vertical_sensitivity = max(0.1, self.vertical_sensitivity - 0.1)
                elif key == ord('a'):
                    self.horizontal_sensitivity = min(5.0, self.horizontal_sensitivity + 0.1)
                elif key == ord('f'):
                    self.horizontal_sensitivity = max(0.1, self.horizontal_sensitivity - 0.1)

        except Exception as e:
            print(f"Error occurred: {str(e)}")
        finally:
            self.cleanup()

    def process_current_key(self):
        # Calculate keyboard layout position
        start_x = (self.window_width - (len(self.keys[0]) * (self.key_width + self.key_spacing))) // 2
        start_y = self.window_height - (len(self.keys) * (self.key_height + self.key_spacing)) - 50

        # Check each key
        for row_idx, row in enumerate(self.keys):
            for col_idx, key in enumerate(row):
                x = start_x + col_idx * (self.key_width + self.key_spacing)
                y = start_y + row_idx * (self.key_height + self.key_spacing)

                width = self.key_width * 2 if row_idx == len(self.keys) - 1 and key in ['SHIFT', 'SPACE', 'BACKSPACE',
                                                                                        'ENTER',
                                                                                        'CLEAR'] else self.key_width

                if (x < self.cursor_x < x + width and
                        y < self.cursor_y < y + self.key_height):
                    self.process_key(key)
                    return

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()



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


class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye Tracking Control Panel")
        self.root.geometry("600x400")
        self.root.configure(bg="#2c3e50")

        # Title Label
        title_label = tk.Label(
            self.root,
            text="Eye Tracking System",
            font=("Helvetica", 24, "bold"),
            fg="white",
            bg="#2c3e50"
        )
        title_label.pack(pady=20)

        # Buttons Frame
        button_frame = tk.Frame(self.root, bg="#2c3e50")
        button_frame.pack(pady=50)

        # Check Cam Button
        cam_button = tk.Button(
            button_frame,
            text="Check Cam",
            font=("Helvetica", 16),
            bg="#27ae60",
            fg="white",
            activebackground="#2ecc71",
            activeforeground="white",
            width=20,
            command=self.open_cam
        )
        cam_button.grid(row=0, column=0, padx=20, pady=10)

        # Objects Button
        objects_button = tk.Button(
            button_frame,
            text="Objects",
            font=("Helvetica", 16),
            bg="#2980b9",
            fg="white",
            activebackground="#3498db",
            activeforeground="white",
            width=20,
            command=self.open_objects
        )
        objects_button.grid(row=1, column=0, padx=20, pady=10)

        # Virtual Keyboard Button
        keyboard_button = tk.Button(
            button_frame,
            text="Virtual Keyboard",
            font=("Helvetica", 16),
            bg="#8e44ad",
            fg="white",
            activebackground="#9b59b6",
            activeforeground="white",
            width=20,
            command=self.open_virtual_keyboard
        )
        keyboard_button.grid(row=2, column=0, padx=20, pady=10)

        # Footer Label
        footer_label = tk.Label(
            self.root,
            text="Select an option to proceed",
            font=("Helvetica", 14),
            fg="white",
            bg="#2c3e50"
        )
        footer_label.pack(pady=10)

    def open_cam(self):
        try:
            tracker = EightDirectionEyeTracker()
            tracker.run()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open cam.py: {e}")

    def open_objects(self):
        try:
            game = EyeTrackingGame()
            game.run()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open object.py: {e}")

    def open_virtual_keyboard(self):
        try:
            keyboard = EyeTrackingKeyboard()
            keyboard.run()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open virtual.py: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
