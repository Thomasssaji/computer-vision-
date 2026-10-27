
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# Project paths
BASE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = BASE_DIR / "reference_images"
TEST_IMAGE = BASE_DIR / "test_images" / "test_apple.jpg"

FRESH_IMAGE = REFERENCE_DIR / "fresh.jpg"
ROTTEN_IMAGE = REFERENCE_DIR / "rotten.jpg"

# Use CPU for compatibility
device = torch.device("cpu")

print("Loading pretrained image model...")

weights = MobileNet_V3_Small_Weights.DEFAULT
model = mobilenet_v3_small(weights=weights)

# Use the feature extractor, excluding the classification head
model.classifier = torch.nn.Identity()
model = model.to(device)
model.eval()

preprocess = weights.transforms()


def extract_features(image_path):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(image_tensor)
        features = F.normalize(features, p=2, dim=1)

    return features


def main():
    required_files = [FRESH_IMAGE, ROTTEN_IMAGE, TEST_IMAGE]

    for file_path in required_files:
        if not file_path.exists():
            print(f"\nERROR: Missing image: {file_path}")
            return

    print("Extracting reference image features...")

    fresh_features = extract_features(FRESH_IMAGE)
    rotten_features = extract_features(ROTTEN_IMAGE)

    print("Analyzing test image...")

    test_features = extract_features(TEST_IMAGE)

    fresh_similarity = F.cosine_similarity(
        test_features, fresh_features
    ).item()

    rotten_similarity = F.cosine_similarity(
        test_features, rotten_features
    ).item()

    print("\n" + "=" * 40)
    print("      FRUIT QUALITY INSPECTION")
    print("=" * 40)

    print(f"Fresh reference similarity:  {fresh_similarity:.4f}")
    print(f"Rotten reference similarity: {rotten_similarity:.4f}")

    if fresh_similarity > rotten_similarity:
        prediction = "FRESH"
    elif rotten_similarity > fresh_similarity:
        prediction = "ROTTEN"
    else:
        prediction = "UNCERTAIN"

    print(f"\nPredicted result: {prediction}")
    print("\nNote: Similarity is not a probability or confidence score.")
    print("Verify the result visually; this prototype may misclassify.")
    print("=" * 40)


if __name__ == "__main__":
    main()

