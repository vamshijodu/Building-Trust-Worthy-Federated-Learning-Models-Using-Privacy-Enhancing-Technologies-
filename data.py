from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch


def load_data(cid, num_clients=2):
    """
    Load MNIST dataset and create a federated shard for the given client.

    Improvements:
    - Proper normalization (critical for DP training stability)
    - Randomized but reproducible partitioning
    - Balanced sharding across clients
    - Smaller batch size (better for DP-SGD)
    """

    # ----------------------------
    # 1️⃣ Proper normalization
    # ----------------------------
    # MNIST mean/std (empirically computed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        "data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        "data",
        train=False,
        download=True,
        transform=transform
    )

    # ----------------------------
    # 2️⃣ Create balanced shards
    # ----------------------------
    # Shuffle indices once (fixed seed for reproducibility)
    np.random.seed(42)
    indices = np.random.permutation(len(train_dataset))

    shard_size = 5000  # increased from 3000 for better learning
    start = cid * shard_size
    end = start + shard_size

    client_indices = indices[start:end]
    train_subset = Subset(train_dataset, client_indices)

    # ----------------------------
    # 3️⃣ DP-friendly DataLoader
    # ----------------------------
    trainloader = DataLoader(
        train_subset,
        batch_size=64,          # smaller batch works better with DP
        shuffle=True,
        num_workers=0
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0
    )

    return trainloader, testloader
