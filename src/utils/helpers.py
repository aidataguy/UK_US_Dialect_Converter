import random
import torch
import numpy as np
import logging

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def calculate_metrics(predictions, targets):
    """Calculate evaluation metrics."""
    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    accuracy = correct / len(predictions)
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(predictions)
    }