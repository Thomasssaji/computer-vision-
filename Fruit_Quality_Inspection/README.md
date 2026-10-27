# Fruit Quality Inspection System

## Overview

This project is a Python-based fruit image inspection prototype that compares a test image against fresh and rotten reference images using image features extracted from a pretrained MobileNetV3-Small model.

## Features

* Compares a test image with two reference images.
* Calculates cosine similarity scores.
* Predicts whether the test image is more similar to the fresh or rotten reference.
* Runs locally using Python and PyTorch.

## Technologies Used

* Python
* PyTorch
* Torchvision
* Pillow
* MobileNetV3-Small

## Project Structure

```text
Fruit_Quality_Inspection/
├── reference_images/
│   ├── fresh.jpg
│   └── rotten.jpg
├── test_images/
│   └── test_apple.jpg
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Create and activate a virtual environment, then install the dependencies.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the Project

Place the fresh and rotten reference images in `reference_images/` and the test image in `test_images/`.

Run:

```bash
python predict.py
```

The first run downloads the pretrained model weights and requires an internet connection.

## How It Works

The program extracts image features using a pretrained MobileNetV3-Small model and calculates cosine similarity between the test image and each reference image. It predicts the category corresponding to the higher similarity score.

## Limitations

This is a similarity-based prototype, not a dedicated fruit-quality classifier. Similarity scores are not probabilities or confidence percentages. Results depend on the reference images and may be inaccurate for different fruit varieties, backgrounds, lighting conditions, and decay patterns.

## Future Improvements

* Use a larger, properly labelled dataset.
* Train or fine-tune a dedicated fruit-quality classifier.
* Evaluate performance on independent test images.
* Add a graphical user interface.
    