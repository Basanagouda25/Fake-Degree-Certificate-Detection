"""
fl_client.py
------------
Flower FL Client for Fake Certificate Detection.

Each client holds a local partition of the training data and performs
local training when instructed by the FL server. Clients never share
raw data — only model weight updates are communicated.
"""

import flwr as fl
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple

from fl_model import create_model, get_model_weights, set_model_weights, IMG_SIZE

# Type aliases
NDArrays = List[np.ndarray]
Scalar = float


class FakeCertClient(fl.client.NumPyClient):
    """
    Flower NumPy client for federated fake certificate detection.

    Each client:
    - Holds its own local partition of training data
    - Shares the same validation set (for consistent evaluation)
    - Trains locally and returns updated weights to the server
    """

    def __init__(
        self,
        client_id: int,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs_per_round: int = 3,
        batch_size: int = 16,
    ):
        super().__init__()
        self.client_id = client_id
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.epochs_per_round = epochs_per_round
        self.batch_size = batch_size

        # Create a fresh model instance for this client
        self.model = create_model()
        print(f"  [Client {self.client_id}] Initialized with "
              f"{len(self.x_train)} training samples, "
              f"{len(self.x_val)} validation samples")

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the current local model weights."""
        return get_model_weights(self.model)

    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Receive global model parameters, train locally, return updated weights.

        Args:
            parameters: Global model weights from the server.
            config: Configuration dict from the server.

        Returns:
            Tuple of (updated_weights, num_training_samples, metrics_dict)
        """
        # Set the global model weights
        set_model_weights(self.model, parameters)

        # Get training config from server (with defaults)
        epochs = int(config.get("epochs", self.epochs_per_round))
        batch_size = int(config.get("batch_size", self.batch_size))

        print(f"\n  [Client {self.client_id}] Training for {epochs} epoch(s) "
              f"on {len(self.x_train)} samples...")

        # Train locally
        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.x_val, self.y_val),
            verbose=1,
        )

        # Extract metrics from the last epoch
        train_loss = history.history["loss"][-1]
        train_acc = history.history["accuracy"][-1]
        val_loss = history.history["val_loss"][-1]
        val_acc = history.history["val_accuracy"][-1]

        print(f"  [Client {self.client_id}] Done — "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Return updated weights, number of samples used, and metrics
        return (
            get_model_weights(self.model),
            len(self.x_train),
            {
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
            }
        )

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the global model on the local validation data.

        Args:
            parameters: Global model weights from the server.
            config: Configuration dict from the server.

        Returns:
            Tuple of (loss, num_val_samples, metrics_dict)
        """
        # Set the global model weights
        set_model_weights(self.model, parameters)

        # Evaluate on local validation data
        loss, accuracy = self.model.evaluate(
            self.x_val, self.y_val,
            batch_size=self.batch_size,
            verbose=0,
        )

        print(f"  [Client {self.client_id}] Evaluation — "
              f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        return (
            float(loss),
            len(self.x_val),
            {"accuracy": float(accuracy)}
        )
