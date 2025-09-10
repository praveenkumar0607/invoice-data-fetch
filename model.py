# src/model.py
import torch.nn as nn
from transformers import LayoutLMv3ForTokenClassification
from .config import ExtractionConfig

class InvoiceExtractor(nn.Module):
    """Main model for invoice key-value extraction"""
    def __init__(self, config: ExtractionConfig):
        super().__init__()
        self.config = config
        self.num_labels = len(config.fields_to_extract)
        
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=self.num_labels
        )
    
    def forward(self, input_ids, attention_mask, bbox, pixel_values, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            labels=labels
        )