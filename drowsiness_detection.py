import cv2
import mediapipe as mp
import os
import time
import math
import winsound
import threading

# ==================================================
# CONFIGURATION
# ==================================================

MODEL_PATH = "face_landmarker.task"

# Initial eye-closure threshold
EAR_THRESHOLD = 0.21

# Duration eyes must remain closed before alert (seconds)
CLOSED_EYE_DURATION = 1.5

# Alarm settings
ALARM_FREQUENCY = 1500
ALARM_DURATION = 500  # milliseconds
ALARM_INTERVAL = 1.0  # seconds between alarm beeps

# ==================================================
# CHECK MODEL FILE
# ==================================================

if not os.path.exists(MODEL_PATH):
    print("ERROR: face_landmarker.task not found.")
    print("Make sure it is in the project folder.")
    raise SystemExit

# ==================================================
# MEDIAPIPE SETUP
# ==================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

# ==================================================
# EYE LANDMARK INDICES
# ==================================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# ==================================================
# ALARM CONTROL
# ==================================================

alarm_active = threading.Event()
alarm_stop = threading.Event()


def alarm_worker():
    """Play repeated beeps without blocking webcam processing."""

    while not alarm_stop.is_set():

        if alarm_active.is_set():

            try:
                winsound.Beep(
                    ALARM_FREQUENCY,
                    ALARM_DURATION
                )
            except RuntimeError:
                pass

            # Wait between beeps; can stop early if alert clears
            alarm_stop.wait(ALARM_INTERVAL)

        else:
            # Wait briefly before checking again
            alarm_stop.wait(0.1)


# ==================================================
# DISTANCE CALCULATION
# ==================================================

def calculate_distance(point1, point2):

    return math.hypot(
        point1[0] - point2[0],
        point1[1] - point2[1]
    )


# ==================================================
# EYE ASPECT RATIO CALCULATION
# ==================================================

def calculate_ear(landmarks, eye_indices, width, height):

    points = []

    for index in eye_indices:

        landmark = landmarks[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    # Vertical eye distances
    vertical_1 = calculate_distance(
        points[1],
        points[5]
    )

    vertical_2 = calculate_distance(
        points[2],
        points[4]
    )

    # Horizontal eye distance
    horizontal = calculate_distance(
        points[0],
        points[3]
    )

    if horizontal == 0:
        return 0.0

    ear = (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)

    return ear


# ==================================================
# START WEBCAM
# ==================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Could not open webcam.")
    raise SystemExit

print("========================================")
print(" DRIVER DROWSINESS DETECTION SYSTEM")
print("========================================")
print("Webcam started successfully.")
print("Press Q to exit.")
print("")

# Eye-closure tracking
eye_closed_start = None

# Timestamp for MediaPipe video frames
start_time = time.monotonic()

# Track when the last alert began
alert_started = False

# Start background alarm thread
alarm_thread = threading.Thread(
    target=alarm_worker,
    daemon=True
)

alarm_thread.start()

# ==================================================
# MAIN DETECTION LOOP
# ==================================================

try:

    with FaceLandmarker.create_from_options(options) as landmarker:

        while True:

            success, frame = camera.read()

            if not success:

                print("ERROR: Could not read webcam frame.")
                break

            # Mirror webcam image
            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Create MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Increasing timestamp in milliseconds
            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            # Detect facial landmarks
            results = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # Default status
            status = "NO FACE DETECTED"
            status_color = (0, 0, 255)

            if results.face_landmarks:

                landmarks = results.face_landmarks[0]

                # Calculate left-eye EAR
                left_ear = calculate_ear(
                    landmarks,
                    LEFT_EYE,
                    width,
                    height
                )

                # Calculate right-eye EAR
                right_ear = calculate_ear(
                    landmarks,
                    RIGHT_EYE,
                    width,
                    height
                )

                # Average both eyes
                average_ear = (
                    left_ear + right_ear
                ) / 2.0

                # Draw eye landmarks
                for eye_indices in [LEFT_EYE, RIGHT_EYE]:

                    for index in eye_indices:

                        landmark = landmarks[index]

                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        cv2.circle(
                            frame,
                            (x, y),
                            3,
                            (0, 255, 255),
                            -1
                        )

                # ----------------------------------
                # EYE-CLOSURE DETECTION
                # ----------------------------------

                if average_ear < EAR_THRESHOLD:

                    status = "EYES CLOSED"
                    status_color = (0, 165, 255)

                    if eye_closed_start is None:

                        eye_closed_start = time.monotonic()

                    closed_duration = (
                        time.monotonic() - eye_closed_start
                    )

                    # Trigger alert after sustained closure
                    if closed_duration >= CLOSED_EYE_DURATION:

                        status = "DROWSINESS ALERT!"
                        status_color = (0, 0, 255)

                        # Start repeated sound alarm
                        alarm_active.set()

                        # Show alert duration
                        cv2.putText(
                            frame,
                            f"Closed: {closed_duration:.1f} s",
                            (20, 175),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )

                        if not alert_started:

                            print(
                                "WARNING: Drowsiness detected!"
                            )

                            alert_started = True

                else:

                    # Eyes are open; reset closure timer
                    eye_closed_start = None

                    # Stop sound alarm
                    alarm_active.clear()

                    alert_started = False

                    status = "NORMAL"
                    status_color = (0, 255, 0)

                # ----------------------------------
                # DISPLAY EYE RATIOS
                # ----------------------------------

                cv2.putText(
                    frame,
                    f"Average EAR: {average_ear:.3f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Left EAR: {left_ear:.3f}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Right EAR: {right_ear:.3f}",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Threshold: {EAR_THRESHOLD:.2f}",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            else:

                # Reset timer and stop alarm if face disappears
                eye_closed_start = None
                alarm_active.clear()
                alert_started = False

            # ----------------------------------
            # DISPLAY STATUS
            # ----------------------------------

            cv2.putText(
                frame,
                status,
                (20, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                status_color,
                2
            )

            # Display webcam window
            cv2.imshow(
                "Driver Drowsiness Detection",
                frame
            )

            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:

    # Stop alarm thread
    alarm_active.clear()
    alarm_stop.set()

    # Release webcam
    camera.release()

    # Close OpenCV windows
    cv2.destroyAllWindows()

    # Wait briefly for alarm thread to finish
    if alarm_thread.is_alive():
        alarm_thread.join(timeout=1.0)

    print("Driver Drowsiness Detection Stopped.")