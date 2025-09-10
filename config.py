# src/config.py
import json
from dataclasses import dataclass, field
from typing import Dict, Tuple, List

@dataclass
class ExtractionConfig:
    """Configuration class for the extraction system"""
    model_name: str = "microsoft/layoutlmv3-base"
    max_seq_length: int = 512
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 500
    output_dir: str = "./models"
    data_dir: str = "./data"
    img_size: Tuple[int, int] = (224, 224)
    
    fields_to_extract: Dict[str, int] = field(default_factory=lambda: {
        'O': 0,
        'B-INVOICE_NUMBER': 1, 'I-INVOICE_NUMBER': 2,
        'B-INVOICE_DATE': 3, 'I-INVOICE_DATE': 4,
        'B-LINE_ITEM': 5, 'I-LINE_ITEM': 6,
        'B-QUANTITY': 7, 'I-QUANTITY': 8,
        'B-PRICE': 9, 'I-PRICE': 10,
        'B-TOTAL': 11, 'I-TOTAL': 12
    })

    @classmethod
    def from_json_file(cls, file_path: str):
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)