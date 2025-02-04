from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    model_name: str = "t5-small"
    max_length: int = 128
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
@dataclass
class DataConfig:
    train_path: str = "data/raw/train.csv"
    val_path: str = "data/raw/val.csv"
    test_path: str = "data/raw/test.csv"
    processed_data_dir: str = "data/processed"
    
@dataclass
class TrainingConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    output_dir: str = "models/trained_model"
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500