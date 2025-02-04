import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration

class DialectTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(config.model_name)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def generate(self, input_ids, attention_mask, max_length):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length
        )