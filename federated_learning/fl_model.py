"""
fl_model.py
-----------
Shared model definition for Federated Learning.
Uses MobileNetV2 (transfer learning) for binary classification
of certificates as Real or Fake.

This module is used by both the FL clients and the FL server.
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, Input, Rescaling,
    RandomFlip, RandomRotation
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# --- Configuration ---
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)
LEARNING_RATE = 0.00002


def create_model() -> tf.keras.Model:
    """
    Create and compile the MobileNetV2-based model for fake certificate detection.
    Same architecture as the original centralized version.

    Returns:
        A compiled tf.keras.Model ready for training.
    """
    # Data augmentation layers (applied during training)
    data_augmentation = Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.1),
    ], name="data_augmentation")

    # Base model: MobileNetV2 pre-trained on ImageNet
    base_model = MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True  # Fine-tune all layers

    # Build the model
    inputs = Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = Rescaling(1.0 / 127.5, offset=-1)(x)  # Normalize to [-1, 1] for MobileNet
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(
        1, activation='sigmoid',
        kernel_regularizer=tf.keras.regularizers.l2(0.005)
    )(x)

    model = Model(inputs, outputs, name="fake_cert_detector")

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=BinaryCrossentropy(label_smoothing=0.0),
        metrics=['accuracy']
    )

    return model


def get_model_weights(model: tf.keras.Model):
    """Extract model weights as a list of NumPy arrays."""
    return model.get_weights()


def set_model_weights(model: tf.keras.Model, weights):
    """Set model weights from a list of NumPy arrays."""
    model.set_weights(weights)


if __name__ == "__main__":
    # Quick test: build model and print summary
    model = create_model()
    model.summary()
    print(f"\nModel created successfully with {len(model.get_weights())} weight arrays.")
