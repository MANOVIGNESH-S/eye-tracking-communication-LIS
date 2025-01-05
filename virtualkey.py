import cv2
import mediapipe as mp
import numpy as np
import time


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

if __name__ == "__main__":
    keyboard = EyeTrackingKeyboard()
    keyboard.run()