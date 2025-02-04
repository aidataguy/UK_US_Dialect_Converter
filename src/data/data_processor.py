import pandas as pd
from typing import Tuple, List
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer

class DialectDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[str], tokenizer: T5Tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        targets = self.tokenizer(
            label,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze()
        }

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = T5Tokenizer.from_pretrained(config.model_name)
        
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # Load data
        train_df = pd.read_csv(self.config.train_path)
        val_df = pd.read_csv(self.config.val_path)
        test_df = pd.read_csv(self.config.test_path)
        
        # Create datasets
        train_dataset = DialectDataset(
            train_df["uk_text"].tolist(),
            train_df["us_text"].tolist(),
            self.tokenizer,
            self.config.max_length
        )
        
        val_dataset = DialectDataset(
            val_df["uk_text"].tolist(),
            val_df["us_text"].tolist(),
            self.tokenizer,
            self.config.max_length
        )
        
        test_dataset = DialectDataset(
            test_df["uk_text"].tolist(),
            test_df["us_text"].tolist(),
            self.tokenizer,
            self.config.max_length
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader 