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

if __name__ == "__main__":
    game = EyeTrackingGame()
    game.run()