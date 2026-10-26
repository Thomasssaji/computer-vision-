# Driver Drowsiness Detection Using Computer Vision

## Project Overview

This project detects potential driver drowsiness using a laptop webcam and computer vision techniques. It analyses facial landmarks and eye movements to identify prolonged eye closure and provides an audible warning.

## Features

* Real-time webcam monitoring
* Facial landmark detection using MediaPipe
* Eye Aspect Ratio (EAR) calculation
* Eye-closure detection
* Drowsiness alert based on prolonged eye closure
* Repeated sound alarm
* Real-time visual status display

## Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy
* Computer Vision

## How It Works

1. The webcam captures live video.
2. MediaPipe detects facial landmarks.
3. Eye landmarks are used to calculate the Eye Aspect Ratio.
4. The program checks whether the eye ratio indicates eye closure.
5. If the eyes remain closed beyond a specified duration, the system displays a drowsiness alert and plays a sound alarm.
6. When the eyes open again, the alert is cleared.

## Requirements

* Python installed on Windows
* A working webcam
* The required Python libraries
* The MediaPipe Face Landmarker model file

## Installation

Clone or download this repository.

Install the required libraries:

```bash
pip install -r requirements.txt
```

Ensure that `face_landmarker.task` is present in the project directory.

## Run the Project

Execute the following command:

```bash
python drowsiness_detection.py
```

Press `Q` in the webcam window to exit.

## Disclaimer

This project is an educational computer vision prototype. It is not a certified driver-monitoring or road-safety system and should not be relied upon for real-world driving decisions.
