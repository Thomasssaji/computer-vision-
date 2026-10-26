import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open the webcam.")
    exit()

print("Webcam opened successfully. Press Q to close.")

while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read a frame from the webcam.")
        break

    cv2.imshow("Driver Drowsiness Detection - Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()