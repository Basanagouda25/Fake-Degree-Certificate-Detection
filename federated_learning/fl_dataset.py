"""
fl_dataset.py
-------------
Dataset loading and partitioning for Federated Learning.

Loads images from dataset_clean/ into NumPy arrays, then partitions
the training data across N simulated FL clients (IID partitioning).
"""

import os
import numpy as np
from tensorflow.keras.preprocessing import image
from typing import List, Tuple

# --- Configuration ---
IMG_SIZE = (224, 224)
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset_clean"
)


def load_images_from_directory(directory: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all images from a directory that has 'real' and 'fake' subdirectories.

    Args:
        directory: Path to directory containing 'real/' and 'fake/' subdirs.

    Returns:
        Tuple of (images_array, labels_array) where:
            - images_array: shape (N, 224, 224, 3), dtype float32
            - labels_array: shape (N, 1), dtype float32
              (0.0 = fake, 1.0 = real)
    """
    images = []
    labels = []

    # Load FAKE images (label = 0)
    fake_dir = os.path.join(directory, "fake")
    if os.path.exists(fake_dir):
        for fname in os.listdir(fake_dir):
            fpath = os.path.join(fake_dir, fname)
            try:
                img = image.load_img(fpath, target_size=IMG_SIZE)
                img_array = image.img_to_array(img)
                images.append(img_array)
                labels.append(0.0)
            except Exception as e:
                print(f"  Warning: Could not load {fpath}: {e}")

    # Load REAL images (label = 1)
    real_dir = os.path.join(directory, "real")
    if os.path.exists(real_dir):
        for fname in os.listdir(real_dir):
            fpath = os.path.join(real_dir, fname)
            try:
                img = image.load_img(fpath, target_size=IMG_SIZE)
                img_array = image.img_to_array(img)
                images.append(img_array)
                labels.append(1.0)
            except Exception as e:
                print(f"  Warning: Could not load {fpath}: {e}")

    images_array = np.array(images, dtype=np.float32)
    labels_array = np.array(labels, dtype=np.float32).reshape(-1, 1)

    return images_array, labels_array


def load_dataset() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load the full training and validation datasets.

    Returns:
        ((x_train, y_train), (x_val, y_val))
    """
    print(f"Loading dataset from: {DATASET_PATH}")

    train_dir = os.path.join(DATASET_PATH, "train")
    val_dir = os.path.join(DATASET_PATH, "val")

    print("  Loading training data...")
    x_train, y_train = load_images_from_directory(train_dir)
    print(f"  Training: {len(x_train)} images ({int(y_train.sum())} real, {len(y_train) - int(y_train.sum())} fake)")

    print("  Loading validation data...")
    x_val, y_val = load_images_from_directory(val_dir)
    print(f"  Validation: {len(x_val)} images ({int(y_val.sum())} real, {len(y_val) - int(y_val.sum())} fake)")

    return (x_train, y_train), (x_val, y_val)


def partition_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_clients: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition training data into non-overlapping subsets for each FL client.
    Uses IID (Independent and Identically Distributed) partitioning:
    data is shuffled and then split evenly.

    Args:
        x_train: Training images array.
        y_train: Training labels array.
        num_clients: Number of FL clients to partition data across.

    Returns:
        List of (x_partition, y_partition) tuples, one per client.
    """
    # Shuffle the data
    num_samples = len(x_train)
    indices = np.random.permutation(num_samples)
    x_shuffled = x_train[indices]
    y_shuffled = y_train[indices]

    # Split into num_clients partitions
    partition_size = num_samples // num_clients
    partitions = []

    for i in range(num_clients):
        start_idx = i * partition_size
        # Last client gets any remaining samples
        if i == num_clients - 1:
            end_idx = num_samples
        else:
            end_idx = start_idx + partition_size

        x_part = x_shuffled[start_idx:end_idx]
        y_part = y_shuffled[start_idx:end_idx]
        partitions.append((x_part, y_part))
        print(f"  Client {i}: {len(x_part)} samples "
              f"({int(y_part.sum())} real, {len(y_part) - int(y_part.sum())} fake)")

    return partitions


if __name__ == "__main__":
    # Quick test: load and partition data
    (x_train, y_train), (x_val, y_val) = load_dataset()
    print(f"\nPartitioning into 3 clients:")
    partitions = partition_data(x_train, y_train, num_clients=3)
    for i, (x, y) in enumerate(partitions):
        print(f"  Client {i}: {x.shape}, {y.shape}")
