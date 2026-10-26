import cv2
import mediapipe as mp
import os

# Model file path
MODEL_PATH = "face_landmarker.task"

# Check whether the model exists
if not os.path.exists(MODEL_PATH):
    print("ERROR: face_landmarker.task not found.")
    print("Place the model in the project folder.")
    raise SystemExit

print("Loading MediaPipe Face Landmarker...")

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit

print("Webcam opened successfully.")
print("Press Q to close the camera.")

try:
    with FaceLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read webcam frame.")
                break

            # Mirror the webcam image
            frame = cv2.flip(frame, 1)

            # Convert image from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Video timestamp in milliseconds
            timestamp_ms = int(
                camera.get(cv2.CAP_PROP_POS_MSEC)
            )

            # Use elapsed wall-clock time to ensure increasing timestamps
            import time
            if "start_time" not in locals():
                start_time = time.monotonic()

            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            # Detect facial landmarks
            results = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if results.face_landmarks:
                height, width, _ = frame.shape

                for face_landmarks in results.face_landmarks:
                    for landmark in face_landmarks:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        cv2.circle(
                            frame,
                            (x, y),
                            1,
                            (0, 255, 0),
                            -1
                        )

                cv2.putText(
                    frame,
                    "FACE DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            else:
                cv2.putText(
                    frame,
                    "NO FACE DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("Driver Drowsiness - Face Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()