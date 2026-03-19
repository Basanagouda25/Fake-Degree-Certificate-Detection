"""
fl_server.py
------------
Flower FL Server configuration for Fake Certificate Detection.

Configures the FedAvg (Federated Averaging) aggregation strategy
and handles saving the final global model after all FL rounds.
"""

import os
import numpy as np
import tensorflow as tf
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters
from typing import List, Optional, Tuple

from fl_model import create_model, get_model_weights, set_model_weights

# Type aliases
NDArrays = List[np.ndarray]

# --- Configuration ---
DEFAULT_NUM_ROUNDS = 5
DEFAULT_MIN_FIT_CLIENTS = 3
DEFAULT_MIN_EVALUATE_CLIENTS = 3
DEFAULT_MIN_AVAILABLE_CLIENTS = 3
LOCAL_EPOCHS_PER_ROUND = 3
LOCAL_BATCH_SIZE = 16
MODEL_SAVE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fl_global_model.h5"
)


def get_initial_parameters() -> fl.common.Parameters:
    """
    Create initial model parameters from a freshly initialized model.
    These are sent to all clients at the start of FL training.
    """
    model = create_model()
    initial_weights = get_model_weights(model)
    return ndarrays_to_parameters(initial_weights)


def create_strategy(
    num_clients: int = DEFAULT_MIN_AVAILABLE_CLIENTS,
    local_epochs: int = LOCAL_EPOCHS_PER_ROUND,
    local_batch_size: int = LOCAL_BATCH_SIZE,
) -> FedAvg:
    """
    Create a FedAvg strategy for the FL server.

    FedAvg works by:
    1. Sending the global model to selected clients
    2. Each client trains locally for a few epochs
    3. Clients send weight updates back to the server
    4. Server averages the weights (weighted by number of samples)
    5. Repeat for N rounds

    Args:
        num_clients: Number of clients required for each round.
        local_epochs: Number of local training epochs per round.
        local_batch_size: Batch size for local training.

    Returns:
        Configured FedAvg strategy.
    """

    def fit_config(server_round: int):
        """Configuration sent to each client during fit."""
        return {
            "epochs": local_epochs,
            "batch_size": local_batch_size,
            "server_round": server_round,
        }

    def evaluate_config(server_round: int):
        """Configuration sent to each client during evaluation."""
        return {
            "server_round": server_round,
        }

    strategy = FedAvg(
        fraction_fit=1.0,           # Use 100% of available clients for training
        fraction_evaluate=1.0,      # Use 100% of available clients for evaluation
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=get_initial_parameters(),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
    )

    return strategy


def save_global_model(parameters: NDArrays, save_path: str = MODEL_SAVE_PATH):
    """
    Save the final aggregated global model to disk.

    Args:
        parameters: The final global model weights (list of NumPy arrays).
        save_path: Path to save the model.
    """
    print(f"\n{'='*60}")
    print(f"Saving global model to: {save_path}")
    print(f"{'='*60}")

    model = create_model()
    set_model_weights(model, parameters)
    model.save(save_path)
    print(f"✅ Global model saved successfully!")
    return save_path
