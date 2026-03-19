"""
fl_predict.py
-------------
Prediction script using the Federated Learning trained global model.

Usage:
    python fl_predict.py <path_to_certificate_image>

Example:
    python fl_predict.py ../dataset_clean/val/fake/some_image.jpg
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# --- Configuration ---
IMG_SIZE = (224, 224)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fl_global_model.h5"
)
THRESHOLD = 0.5  # Score >= 0.5 → REAL, < 0.5 → FAKE


def load_and_predict(model_path: str, img_path: str):
    """
    Load the FL-trained global model and predict if a certificate is real or fake.

    Args:
        model_path: Path to the saved global model (.h5 file).
        img_path: Path to the certificate image to classify.
    """
    # Validate paths
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at '{model_path}'")
        print("   Please run fl_simulation.py first to train the model.")
        return

    if not os.path.exists(img_path):
        print(f"❌ Error: Image not found at '{img_path}'")
        return

    # Load model
    print(f"Loading FL global model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess image
    print(f"Processing image: {img_path}")
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array, verbose=0)
    score = float(prediction[0][0])

    # Interpret result
    if score >= 0.7:
        label = "REAL"
        confidence = score
    elif score <= 0.3:
        label = "FAKE"
        confidence = 1.0 - score
    else:
        label = "UNCERTAIN"
        confidence = max(score, 1.0 - score)

    # Display results
    print(f"\n{'='*50}")
    print(f"  PREDICTION RESULTS (Federated Learning Model)")
    print(f"{'='*50}")
    print(f"  Image:            {os.path.basename(img_path)}")
    print(f"  Raw Score:        {score:.4f}")
    print(f"  Predicted Label:  {label}")
    print(f"  Confidence:       {confidence:.2%}")
    print(f"{'='*50}")

    return label, score


def main():
    if len(sys.argv) < 2:
        print("Usage: python fl_predict.py <path_to_certificate_image>")
        print("\nExample:")
        print("  python fl_predict.py ../dataset_clean/val/fake/some_image.jpg")

        # If no argument provided, try to find a sample image
        val_fake_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dataset_clean", "val", "fake"
        )
        val_real_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dataset_clean", "val", "real"
        )

        # Auto-demo: test one fake and one real image if available
        if os.path.exists(MODEL_PATH):
            print(f"\n--- Auto-Demo (using available validation images) ---\n")

            # Test a fake image
            if os.path.exists(val_fake_dir):
                fake_images = os.listdir(val_fake_dir)
                if fake_images:
                    fake_path = os.path.join(val_fake_dir, fake_images[0])
                    print(f"Testing FAKE certificate:")
                    load_and_predict(MODEL_PATH, fake_path)

            # Test a real image
            if os.path.exists(val_real_dir):
                real_images = os.listdir(val_real_dir)
                if real_images:
                    real_path = os.path.join(val_real_dir, real_images[0])
                    print(f"\nTesting REAL certificate:")
                    load_and_predict(MODEL_PATH, real_path)
        else:
            print(f"\n⚠️  No trained model found at {MODEL_PATH}")
            print("   Run fl_simulation.py first to train the model.")
        return

    img_path = sys.argv[1]
    load_and_predict(MODEL_PATH, img_path)


if __name__ == "__main__":
    main()
