import tkinter as tk
from tkinter import messagebox
import subprocess
import cv2
import mediapipe as mp
import numpy as np
import json
import os
from PIL import Image, ImageTk
import time


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
        self.window_name = 'EyeTalk: Communication Aid'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)

        # Cursor parameters - use global configuration instead of local values
        self.cursor_x = self.window_width // 2
        self.cursor_y = self.window_height // 2

        # Don't need these local variables anymore, use global_config instead
        # self.movement_unit = 10
        # self.vertical_sensitivity = 1.0
        # self.horizontal_sensitivity = 1.0
        # self.smoothing = 0.2

        # Communication targets
        self.messages = [
            {"text": "I'm hungry", "pos": (50, 50), "hover_time": 0, "color": (0, 150, 255)},
            {"text": "I'm thirsty", "pos": (self.window_width - 250, 50), "hover_time": 0, "color": (0, 255, 150)},
            {"text": "Restroom", "pos": (50, self.window_height - 100), "hover_time": 0, "color": (255, 100, 0)},
            {"text": "Emergency", "pos": (self.window_width - 250, self.window_height - 100), "hover_time": 0,
             "color": (255, 50, 50)}
        ]

        # Initialize text bounding boxes
        for msg in self.messages:
            self.init_text_object(msg)

        # Dialog management
        self.selected_message = None
        self.close_button_bbox = None
        self.close_button_hover_time = 0

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Calibration parameters
        self.center_left_eye = None
        self.center_right_eye = None
        self.calibrated = False

        # Smooth movement tracking
        self.moving_avg_x = self.window_width // 2
        self.moving_avg_y = self.window_height // 2

        # Eye landmarks
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]

        # Create canvas
        self.canvas = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)

        # Load global configuration at initialization
        try:
            load_global_configuration()
            print("Global configuration loaded for EyeTrackingGame")
        except NameError:
            # If the function is not defined (we're in the same file)
            print("Using already loaded global configuration")

    def init_text_object(self, msg):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(msg["text"], font, font_scale, thickness)

        # Calculate text position and bounding box
        x, y = msg["pos"]
        padding = 15
        msg.update({
            "font": font,
            "font_scale": font_scale,
            "thickness": thickness,
            "text_width": text_width,
            "text_height": text_height,
            "bbox": (
                x - padding,
                y - padding,
                x + text_width + padding,
                y + text_height + padding
            )
        })

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
        print(f"Calibration complete! Using movement unit: {global_config['movement_unit']}")

    def calculate_movement(self, left_center, right_center):
        if not self.calibrated:
            return 0, 0

        left_dev_x = left_center[0] - self.center_left_eye[0]
        right_dev_x = right_center[0] - self.center_right_eye[0]
        left_dev_y = left_center[1] - self.center_left_eye[1]
        right_dev_y = right_center[1] - self.center_right_eye[1]

        # Use global configuration values
        dx = ((left_dev_x + right_dev_x) / 2) * global_config['movement_unit'] * global_config['horizontal_sensitivity']
        dy = ((left_dev_y + right_dev_y) / 2) * global_config['movement_unit'] * global_config['vertical_sensitivity']

        return dx, dy

    def smooth_movement(self, x, y):
        # Use global smoothing factor
        self.moving_avg_x = (self.moving_avg_x * (1 - global_config['smoothing']) +
                             x * global_config['smoothing'])
        self.moving_avg_y = (self.moving_avg_y * (1 - global_config['smoothing']) +
                             y * global_config['smoothing'])
        return self.moving_avg_x, self.moving_avg_y

    def check_interactions(self):
        if self.selected_message:
            # Check close button interaction
            if self.close_button_bbox:
                x1, y1, x2, y2 = self.close_button_bbox
                if (x1 <= self.cursor_x <= x2) and (y1 <= self.cursor_y <= y2):
                    self.close_button_hover_time += 1
                    if self.close_button_hover_time >= 15:  # 0.5 seconds at 30 FPS
                        self.selected_message = None
                        self.close_button_hover_time = 0
                else:
                    self.close_button_hover_time = 0
            return

        # Check text box interactions
        for msg in self.messages:
            x1, y1, x2, y2 = msg["bbox"]
            if (x1 <= self.cursor_x <= x2) and (y1 <= self.cursor_y <= y2):
                msg["hover_time"] += 1
                if msg["hover_time"] >= 30:  # 1 second at 30 FPS
                    self.selected_message = msg["text"]
                    msg["hover_time"] = 0
            else:
                msg["hover_time"] = 0

    def draw_interface(self):
        self.canvas.fill(0)

        # Draw message boxes with hover effects
        for msg in self.messages:
            x1, y1, x2, y2 = msg["bbox"]
            color = msg["color"]
            hover_progress = min(msg["hover_time"] / 30, 1.0)  # Normalize hover time

            # Draw background with hover effect
            bg_color = tuple(int(c * (0.5 + 0.5 * hover_progress)) for c in color)
            cv2.rectangle(self.canvas, (x1, y1), (x2, y2), bg_color, -1, cv2.LINE_AA)
            cv2.rectangle(self.canvas, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)

            # Draw text
            text_x = x1 + 10
            text_y = y2 - 10
            cv2.putText(self.canvas, msg["text"], (text_x, text_y),
                        msg["font"], msg["font_scale"], (255, 255, 255), msg["thickness"], cv2.LINE_AA)

            # Draw progress bar
            if hover_progress > 0:
                bar_height = 5
                bar_width = int((x2 - x1) * hover_progress)
                cv2.rectangle(self.canvas, (x1, y2), (x1 + bar_width, y2 + bar_height), (255, 255, 255), -1)

        # Draw dialog if message selected
        if self.selected_message:
            # Darken background
            overlay = self.canvas.copy()
            cv2.rectangle(overlay, (0, 0), (self.window_width, self.window_height), (0, 0, 0, 0.7), -1)
            cv2.addWeighted(overlay, 0.7, self.canvas, 0.3, 0, self.canvas)

            # Draw dialog box
            dialog_w, dialog_h = 600, 200
            x = self.window_width // 2 - dialog_w // 2
            y = self.window_height // 2 - dialog_h // 2
            cv2.rectangle(self.canvas, (x, y), (x + dialog_w, y + dialog_h), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.rectangle(self.canvas, (x, y), (x + dialog_w, y + dialog_h), (0, 0, 0), 2, cv2.LINE_AA)

            # Draw message
            text_size = cv2.getTextSize(self.selected_message, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            text_x = x + (dialog_w - text_size[0]) // 2
            text_y = y + (dialog_h + text_size[1]) // 2
            cv2.putText(self.canvas, self.selected_message, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)

            # Draw close button
            btn_size = 30
            btn_x = x + dialog_w - btn_size - 20
            btn_y = y + 20
            self.close_button_bbox = (btn_x, btn_y, btn_x + btn_size, btn_y + btn_size)
            cv2.rectangle(self.canvas, (btn_x, btn_y), (btn_x + btn_size, btn_y + btn_size), (0, 0, 255), -1,
                          cv2.LINE_AA)
            cv2.line(self.canvas, (btn_x, btn_y), (btn_x + btn_size, btn_y + btn_size), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(self.canvas, (btn_x + btn_size, btn_y), (btn_x, btn_y + btn_size), (255, 255, 255), 2, cv2.LINE_AA)

        # Draw cursor
        cursor_size = 20
        cv2.line(self.canvas, (int(self.cursor_x - cursor_size), int(self.cursor_y)),
                 (int(self.cursor_x + cursor_size), int(self.cursor_y)), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(self.canvas, (int(self.cursor_x), int(self.cursor_y - cursor_size)),
                 (int(self.cursor_x), int(self.cursor_y + cursor_size)), (0, 255, 0), 2, cv2.LINE_AA)

        # Draw calibration instructions
        if not self.calibrated:
            cv2.putText(self.canvas, "Look straight ahead and press 'c' to calibrate",
                        (self.window_width // 2 - 250, self.window_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Display current configuration values
        cv2.putText(self.canvas, f"Movement Unit: {global_config['movement_unit']}",
                    (10, self.window_height - 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(self.canvas, f"Vertical Sensitivity: {global_config['vertical_sensitivity']:.2f}",
                    (10, self.window_height - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(self.canvas, f"Horizontal Sensitivity: {global_config['horizontal_sensitivity']:.2f}",
                    (10, self.window_height - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(self.canvas, f"Smoothing: {global_config['smoothing']:.2f}",
                    (10, self.window_height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        return self.canvas

    def run(self):
        print("Starting EyeTalk: Communication Aid...")
        print(f"Using global movement unit: {global_config['movement_unit']}")
        print(f"Using global vertical sensitivity: {global_config['vertical_sensitivity']}")
        print(f"Using global horizontal sensitivity: {global_config['horizontal_sensitivity']}")
        print(f"Using global smoothing: {global_config['smoothing']}")
        print("Calibrate first by looking straight ahead and pressing 'c'")
        print("Move your eyes to control the cursor")
        print("Dwell on a message for 1 second to select it")
        print("Press 'q' to quit")
        print("Press 's' to save current configuration")

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
                        self.cursor_x = int(np.clip(self.window_width // 2 + smooth_x, 0, self.window_width))
                        self.cursor_y = int(np.clip(self.window_height // 2 + smooth_y, 0, self.window_height))

                    self.check_interactions()

                # Draw interface and display
                cv2.imshow(self.window_name, self.draw_interface())

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c') and results.multi_face_landmarks:
                    self.calibrate(left_center, right_center)
                # Add these keys for adjusting configuration directly from the game
                elif key == ord('i'):
                    global_config['movement_unit'] = min(50, global_config['movement_unit'] + 1)
                    print(f"Movement unit increased to: {global_config['movement_unit']}")
                elif key == ord('d'):
                    global_config['movement_unit'] = max(1, global_config['movement_unit'] - 1)
                    print(f"Movement unit decreased to: {global_config['movement_unit']}")
                elif key == ord('w'):
                    global_config['vertical_sensitivity'] = min(5.0, global_config['vertical_sensitivity'] + 0.1)
                    print(f"Vertical sensitivity increased to: {global_config['vertical_sensitivity']:.2f}")
                elif key == ord('s'):
                    try:
                        save_global_configuration()  # Save configuration when 's' is pressed
                    except NameError:
                        print("Warning: save_global_configuration function not available")
                elif key == ord('e'):
                    global_config['vertical_sensitivity'] = max(0.1, global_config['vertical_sensitivity'] - 0.1)
                    print(f"Vertical sensitivity decreased to: {global_config['vertical_sensitivity']:.2f}")
                elif key == ord('a'):
                    global_config['horizontal_sensitivity'] = min(5.0, global_config['horizontal_sensitivity'] + 0.1)
                    print(f"Horizontal sensitivity increased to: {global_config['horizontal_sensitivity']:.2f}")
                elif key == ord('f'):
                    global_config['horizontal_sensitivity'] = max(0.1, global_config['horizontal_sensitivity'] - 0.1)
                    print(f"Horizontal sensitivity decreased to: {global_config['horizontal_sensitivity']:.2f}")
                elif key == ord('z'):
                    global_config['smoothing'] = min(0.9, global_config['smoothing'] + 0.05)
                    print(f"Smoothing increased to: {global_config['smoothing']:.2f}")
                elif key == ord('x'):
                    global_config['smoothing'] = max(0.05, global_config['smoothing'] - 0.05)
                    print(f"Smoothing decreased to: {global_config['smoothing']:.2f}")

        finally:
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

        # Cursor parameters - use global configuration
        self.cursor_x = self.window_width // 2
        self.cursor_y = self.window_height // 2

        # Use smoothing from global configuration
        self.smoothing = global_config['smoothing']

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
        print(f"Calibration complete - Movement unit: {global_config['movement_unit']} pixels")
        print("Look at keys and hold gaze or blink to select")

    def calculate_movement(self, left_center, right_center):
        if not self.calibrated:
            return 0, 0

        # Calculate deviations
        left_dev_x = left_center[0] - self.center_left_eye[0]
        right_dev_x = right_center[0] - self.center_right_eye[0]
        left_dev_y = left_center[1] - self.center_left_eye[1]
        right_dev_y = right_center[1] - self.center_right_eye[1]

        # Average deviation scaled by movement unit and sensitivities from global config
        dx = ((left_dev_x + right_dev_x) / 2) * global_config['movement_unit'] * global_config['horizontal_sensitivity']
        dy = ((left_dev_y + right_dev_y) / 2) * global_config['movement_unit'] * global_config['vertical_sensitivity']

        return dx, dy

    def smooth_movement(self, x, y):
        self.moving_avg_x = (self.moving_avg_x * (1 - global_config['smoothing']) +
                             x * global_config['smoothing'])
        self.moving_avg_y = (self.moving_avg_y * (1 - global_config['smoothing']) +
                             y * global_config['smoothing'])
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

        # Draw controls info using global config values
        controls = [
            "Controls:",
            "i/d: increase/decrease movement speed",
            "w/e: increase/decrease vertical sensitivity",
            "a/f: increase/decrease horizontal sensitivity",
            "s: save configuration",
            "q: quit",
            f"Movement Speed: {global_config['movement_unit']}",
            f"V-Sensitivity: {global_config['vertical_sensitivity']:.1f}",
            f"H-Sensitivity: {global_config['horizontal_sensitivity']:.1f}"
        ]

        for i, text in enumerate(controls):
            cv2.putText(frame, text, (10, 200 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    def run(self):
        print("Starting Eye-Tracking Keyboard...")
        print("Look at the center blue dot and press 'c' to calibrate")
        print("Hold gaze for 3 seconds or blink to select a key")
        print("Press 'i' to increase movement unit")
        print("Press 'd' to decrease movement unit")
        print("Press 'w' to increase vertical sensitivity")
        print("Press 'e' to decrease vertical sensitivity")
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
                    global_config['movement_unit'] = min(50, global_config['movement_unit'] + 1)
                    print(f"Movement unit increased to: {global_config['movement_unit']}")
                elif key == ord('d'):
                    global_config['movement_unit'] = max(1, global_config['movement_unit'] - 1)
                    print(f"Movement unit decreased to: {global_config['movement_unit']}")
                elif key == ord('w'):
                    global_config['vertical_sensitivity'] = min(5.0, global_config['vertical_sensitivity'] + 0.1)
                    print(f"Vertical sensitivity increased to: {global_config['vertical_sensitivity']:.2f}")
                elif key == ord('e'):
                    global_config['vertical_sensitivity'] = max(0.1, global_config['vertical_sensitivity'] - 0.1)
                    print(f"Vertical sensitivity decreased to: {global_config['vertical_sensitivity']:.2f}")
                elif key == ord('a'):
                    global_config['horizontal_sensitivity'] = min(5.0, global_config['horizontal_sensitivity'] + 0.1)
                    print(f"Horizontal sensitivity increased to: {global_config['horizontal_sensitivity']:.2f}")
                elif key == ord('f'):
                    global_config['horizontal_sensitivity'] = max(0.1, global_config['horizontal_sensitivity'] - 0.1)
                    print(f"Horizontal sensitivity decreased to: {global_config['horizontal_sensitivity']:.2f}")
                elif key == ord('s'):
                    save_global_configuration()  # Save configuration when 's' is pressed
                    print("Configuration saved successfully.")

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


# Global configuration dictionary that can be accessed by other classes
global_config = {
    "movement_unit": 10,
    "vertical_sensitivity": 1.0,
    "horizontal_sensitivity": 1.0,
    "smoothing": 0.2
}


def save_global_configuration():
    """
    Save the global configuration to a file that persists across program runs
    """
    config_file = "eye_tracker_config.json"
    with open(config_file, "w") as f:
        json.dump(global_config, f, indent=4)
    print(f"Global configuration saved to {config_file}.")


def load_global_configuration():
    """
    Load the global configuration from file
    """
    config_file = "eye_tracker_config.json"
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            loaded_config = json.load(f)
            # Update the global config with loaded values
            global_config.update(loaded_config)
            print(f"Global configuration loaded from {config_file}.")
    else:
        print("No configuration file found. Using default values.")


# Load configuration at module import time
load_global_configuration()


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

        # Cursor parameters - use global configuration
        self.cursor_x = self.window_width // 2
        self.cursor_y = self.window_height // 2

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

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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
        print(f"Calibrated - Movement unit: {global_config['movement_unit']} pixels")

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

        # Average deviation scaled by movement unit and sensitivities from global config
        dx = ((left_dev_x + right_dev_x) / 2) * global_config['movement_unit'] * global_config['horizontal_sensitivity']
        dy = ((left_dev_y + right_dev_y) / 2) * global_config['movement_unit'] * global_config['vertical_sensitivity']

        return dx, dy

    def smooth_movement(self, x, y):
        self.moving_avg_x = (self.moving_avg_x * (1 - global_config['smoothing']) +
                             x * global_config['smoothing'])
        self.moving_avg_y = (self.moving_avg_y * (1 - global_config['smoothing']) +
                             y * global_config['smoothing'])
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

        # Display information using global config values
        cv2.putText(frame, f"Movement Unit: {global_config['movement_unit']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Direction: {direction}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Vertical Sensitivity: {global_config['vertical_sensitivity']:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Horizontal Sensitivity: {global_config['horizontal_sensitivity']:.2f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "i/d: increase/decrease movement", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "w/e: increase/decrease vertical sensitivity", (10, 180),
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
        print("Press 'e' to decrease vertical sensitivity")
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
                        direction = self.get_direction(dx / global_config['movement_unit'],
                                                       dy / global_config['movement_unit'])

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
                    global_config['movement_unit'] = min(50, global_config['movement_unit'] + 1)
                    print(f"Movement unit increased to: {global_config['movement_unit']}")
                elif key == ord('d'):
                    global_config['movement_unit'] = max(1, global_config['movement_unit'] - 1)
                    print(f"Movement unit decreased to: {global_config['movement_unit']}")
                elif key == ord('w'):
                    global_config['vertical_sensitivity'] = min(5.0, global_config['vertical_sensitivity'] + 0.1)
                    print(f"Vertical sensitivity increased to: {global_config['vertical_sensitivity']:.2f}")
                elif key == ord('s'):
                    save_global_configuration()  # Save configuration when 's' is pressed
                elif key == ord('e'):
                    global_config['vertical_sensitivity'] = max(0.1, global_config['vertical_sensitivity'] - 0.1)
                    print(f"Vertical sensitivity decreased to: {global_config['vertical_sensitivity']:.2f}")
                elif key == ord('a'):
                    global_config['horizontal_sensitivity'] = min(5.0, global_config['horizontal_sensitivity'] + 0.1)
                    print(f"Horizontal sensitivity increased to: {global_config['horizontal_sensitivity']:.2f}")
                elif key == ord('f'):
                    global_config['horizontal_sensitivity'] = max(0.1, global_config['horizontal_sensitivity'] - 0.1)
                    print(f"Horizontal sensitivity decreased to: {global_config['horizontal_sensitivity']:.2f}")

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
        self.root.geometry("800x600")  # Adjust as needed
        self.root.state("zoomed")  # Make window maximized without full screen

        # Load and display the background image
        self.bg_image = Image.open("img.png")  # Ensure img.png is in the same directory
        self.bg_image = self.bg_image.resize((self.root.winfo_screenwidth(), self.root.winfo_screenheight()))
        self.bg_image_tk = ImageTk.PhotoImage(self.bg_image)

        # Create a Canvas and place the background image
        self.canvas = tk.Canvas(self.root, width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, image=self.bg_image_tk, anchor="nw")

        # Title Label
        title_label = tk.Label(
            self.root,
            text="Eye Tracking System",
            font=("Helvetica", 24, "bold"),
            fg="white",
            bg="black"
        )
        title_label.place(relx=0.5, rely=0.1, anchor="center")

        # Buttons Frame
        button_frame = tk.Frame(self.root, bg="black")
        button_frame.place(relx=0.5, rely=0.5, anchor="center")

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
            bg="black"
        )
        footer_label.place(relx=0.5, rely=0.9, anchor="center")

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
