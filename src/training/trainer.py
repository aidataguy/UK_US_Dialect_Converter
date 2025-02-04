import torch
from tqdm import tqdm
import logging
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from ..utils.helpers import set_seed

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=len(train_loader) * config.num_epochs
        )
        
        set_seed(config.seed)
        
    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            train_loss = 0
            
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            avg_train_loss = train_loss / len(self.train_loader)
            val_loss = self.evaluate()
            
            logging.info(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()
    
    def evaluate(self):
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
        
        return val_loss / len(self.val_loader)
    
    def save_model(self):
        torch.save(self.model.state_dict(), f"{self.config.output_dir}/best_model.pt")