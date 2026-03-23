"""
fl_simulation.py
-----------------
Main entry point for running the Federated Learning simulation.

Implements a manual Federated Learning loop that simulates the
full FL workflow:
  1. Server initializes the global model
  2. For each round:
     a. Server sends global weights to all clients
     b. Each client trains locally on its own data partition
     c. Clients return updated weights + metrics to server
     d. Server aggregates weights using FedAvg (weighted average)
  3. Server saves the final global model

This approach clearly demonstrates the FL concepts of:
  - Data never leaving the client
  - Only model updates being communicated
  - FedAvg aggregation on the server side

Usage:
    cd c:\\Users\\basan\\Desktop\\project\\Minor_Project
    python federated_learning/fl_simulation.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fl_model import create_model, get_model_weights, set_model_weights
from fl_dataset import load_dataset, partition_data

# ============================================================
#   FL Simulation Configuration
# ============================================================
NUM_CLIENTS = 3       # Number of simulated FL clients
NUM_ROUNDS = 5        # Number of federated learning rounds
EPOCHS_PER_ROUND = 3  # Local training epochs per client per round
BATCH_SIZE = 16       # Local batch size for client training

MODEL_SAVE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fl_global_model.h5"
)


# ============================================================
#   Flower-style FL Client (local training)
# ============================================================
class FLClient:
    """
    Simulated Federated Learning client.
    Each client holds a local data partition and trains the model locally.
    Only model weights are shared — raw data never leaves the client.
    """

    def __init__(self, client_id, x_train, y_train, x_val, y_val):
        self.client_id = client_id
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.model = create_model()
        print(f"  [Client {self.client_id}] Initialized | "
              f"Train: {len(x_train)} samples | Val: {len(x_val)} samples")

    def get_parameters(self):
        """Return current model weights (sent to server)."""
        return get_model_weights(self.model)

    def set_parameters(self, weights):
        """Receive and apply global model weights from server."""
        set_model_weights(self.model, weights)

    # This is the core of FL — local training on private data (of client)
    def fit(self, global_weights, epochs, batch_size):
        """
        Receive global weights, train locally, return updated weights.
        This is the core of FL — local training on private data.
        """
        # Step 1: Apply global model weights
        self.set_parameters(global_weights)

        # Step 2: Train locally on this client's private data
        print(f"\n  [Client {self.client_id}] Training for {epochs} epoch(s) "
              f"on {len(self.x_train)} local samples...")

        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.x_val, self.y_val),
            verbose=1,
        )

        # Step 3: Return updated weights + training metrics
        train_loss = history.history["loss"][-1]
        train_acc = history.history["accuracy"][-1]
        val_loss = history.history["val_loss"][-1]
        val_acc = history.history["val_accuracy"][-1]

        print(f"  [Client {self.client_id}] Done | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        return (
            self.get_parameters(),       # Updated weights
            len(self.x_train),           # Number of samples (for weighted avg)
            {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

    def evaluate(self, global_weights):
        """Evaluate the global model on local validation data."""
        self.set_parameters(global_weights)
        loss, accuracy = self.model.evaluate(
            self.x_val, self.y_val, batch_size=BATCH_SIZE, verbose=0
        )
        return loss, accuracy


# ============================================================
#   FedAvg Aggregation (Server-side)
# ============================================================
def federated_averaging(client_results):
    """
    FedAvg: Aggregate model weights from all clients using weighted average.

    Each client's contribution is weighted by the number of samples it trained on.
    This is the standard aggregation strategy in federated learning.

    Args:
        client_results: List of (weights, num_samples, metrics) from each client.

    Returns:
        Aggregated global model weights.
    """
    # Total number of training samples across all clients
    total_samples = sum(num_samples for _, num_samples, _ in client_results)

    # Initialize aggregated weights with zeros (same structure as first client)
    aggregated_weights = [
        np.zeros_like(w) for w in client_results[0][0]
    ]

    # Weighted sum of all client weights
    for client_weights, num_samples, _ in client_results:
        weight_factor = num_samples / total_samples
        for i, w in enumerate(client_weights):
            aggregated_weights[i] += w * weight_factor

    return aggregated_weights


# ============================================================
#   Plotting
# ============================================================
def plot_fl_metrics(round_metrics, save_path=None):
    """Plot FL training metrics across all rounds."""
    rounds = list(range(1, len(round_metrics) + 1))

    # Extract per-round aggregated metrics
    avg_train_losses = []
    avg_train_accs = []
    avg_val_losses = []
    avg_val_accs = []

    for rm in round_metrics:
        client_metrics = rm["client_metrics"]
        n_clients = len(client_metrics)

        avg_train_losses.append(
            sum(m["train_loss"] for m in client_metrics) / n_clients)
        avg_train_accs.append(
            sum(m["train_accuracy"] for m in client_metrics) / n_clients)
        avg_val_losses.append(
            sum(m["val_loss"] for m in client_metrics) / n_clients)
        avg_val_accs.append(
            sum(m["val_accuracy"] for m in client_metrics) / n_clients)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Federated Learning Training Progress (FedAvg)", fontsize=14, fontweight='bold')

    # Loss
    axes[0].plot(rounds, avg_train_losses, 'b-o', linewidth=2, markersize=8, label='Avg Train Loss')
    axes[0].plot(rounds, avg_val_losses, 'r-s', linewidth=2, markersize=8, label='Avg Val Loss')
    axes[0].set_xlabel("FL Round", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Loss per FL Round", fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(rounds, avg_train_accs, 'b-o', linewidth=2, markersize=8, label='Avg Train Accuracy')
    axes[1].plot(rounds, avg_val_accs, 'r-s', linewidth=2, markersize=8, label='Avg Val Accuracy')
    axes[1].set_xlabel("FL Round", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Accuracy per FL Round", fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "fl_training_metrics.png"
        )
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Training metrics plot saved to: {save_path}")
    plt.show()


# ============================================================
#   Main FL Simulation
# ============================================================
def main():
    print("=" * 60)
    print("  FEDERATED LEARNING SIMULATION")
    print("  Fake Certificate Detection with Flower + MobileNetV2")
    print("=" * 60)
    print(f"\n  Configuration:")
    print(f"    • Number of clients:     {NUM_CLIENTS}")
    print(f"    • Number of FL rounds:   {NUM_ROUNDS}")
    print(f"    • Local epochs/round:    {EPOCHS_PER_ROUND}")
    print(f"    • Local batch size:      {BATCH_SIZE}")
    print(f"    • Aggregation strategy:  FedAvg (Federated Averaging)")
    print()

    # ---- Step 1: Load dataset ----
    print("=" * 60)
    print("STEP 1: Loading dataset")
    print("=" * 60)
    (x_train, y_train), (x_val, y_val) = load_dataset()

    # ---- Step 2: Partition data across clients ----
    print(f"\n{'=' * 60}")
    print(f"STEP 2: Partitioning data across {NUM_CLIENTS} clients (IID)")
    print("=" * 60)
    partitions = partition_data(x_train, y_train, NUM_CLIENTS)

    # ---- Step 3: Initialize clients ----
    print(f"\n{'=' * 60}")
    print(f"STEP 3: Initializing {NUM_CLIENTS} FL clients")
    print("=" * 60)
    clients = []
    for i in range(NUM_CLIENTS):
        x_part, y_part = partitions[i]
        client = FLClient(
            client_id=i,
            x_train=x_part,
            y_train=y_part,
            x_val=x_val,
            y_val=y_val,
        )
        clients.append(client)

    # ---- Step 4: Initialize global model (server) ----
    print(f"\n{'=' * 60}")
    print("STEP 4: Initializing global model on server")
    print("=" * 60)
    global_model = create_model()
    global_weights = get_model_weights(global_model)
    print("  Global model initialized with random weights.")

    # ---- Step 5: Federated Learning rounds ----
    round_metrics = []

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n{'='*60}")
        print(f"  FL ROUND {round_num}/{NUM_ROUNDS}")
        print(f"{'='*60}")

        # 5a. Each client trains locally with global weights
        client_results = []
        for client in clients:
            updated_weights, num_samples, metrics = client.fit(
                global_weights=global_weights,
                epochs=EPOCHS_PER_ROUND,
                batch_size=BATCH_SIZE,
            )
            client_results.append((updated_weights, num_samples, metrics))

        # 5b. Server aggregates weights using FedAvg
        print(f"\n  [Server] Aggregating weights from {len(clients)} clients (FedAvg)...")
        global_weights = federated_averaging(client_results)

        # 5c. Evaluate global model
        set_model_weights(global_model, global_weights)
        global_loss, global_acc = global_model.evaluate(
            x_val, y_val, batch_size=BATCH_SIZE, verbose=0
        )

        # Store metrics
        client_metrics_list = [m for _, _, m in client_results]
        round_metrics.append({
            "round": round_num,
            "global_loss": global_loss,
            "global_accuracy": global_acc,
            "client_metrics": client_metrics_list,
        })

        print(f"\n  [Server] Round {round_num} Results:")
        print(f"    Global Val Loss:     {global_loss:.4f}")
        print(f"    Global Val Accuracy: {global_acc:.4f}")
        for i, m in enumerate(client_metrics_list):
            print(f"    Client {i} — Train Acc: {m['train_accuracy']:.4f}, "
                  f"Val Acc: {m['val_accuracy']:.4f}")

    # ---- Step 6: Save global model ----
    print(f"\n{'='*60}")
    print("STEP 6: Saving final global model")
    print("=" * 60)
    set_model_weights(global_model, global_weights)
    global_model.save(MODEL_SAVE_PATH)
    print(f"  ✅ Global model saved to: {MODEL_SAVE_PATH}")

    # ---- Step 7: Final evaluation ----
    print(f"\n{'='*60}")
    print("STEP 7: Final evaluation of the global model")
    print("=" * 60)
    final_loss, final_acc = global_model.evaluate(x_val, y_val, verbose=0)
    print(f"  Final Validation Loss:     {final_loss:.4f}")
    print(f"  Final Validation Accuracy: {final_acc:.4f}")

    # ---- Step 8: Plot metrics ----
    print(f"\n{'='*60}")
    print("STEP 8: Plotting training metrics")
    print("=" * 60)
    plot_fl_metrics(round_metrics)

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("  ✅ FEDERATED LEARNING SIMULATION COMPLETE!")
    print(f"{'='*60}")
    print(f"  • {NUM_CLIENTS} clients trained over {NUM_ROUNDS} FL rounds")
    print(f"  • Aggregation: FedAvg (Federated Averaging)")
    print(f"  • Final Accuracy: {final_acc:.4f}")
    print(f"  • Model saved at: {MODEL_SAVE_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
