# src/inference.py
import os
import torch
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from transformers import LayoutLMv3Processor
from .config import ExtractionConfig
from .model import InvoiceExtractor
from .dataset import DocumentOCR
from typing import Dict, List

class InferenceEngine:
    """Inference pipeline for extracting key-value pairs from invoices"""
    def __init__(self, model_path: str, config: ExtractionConfig):
        self.config = config
        self.processor = LayoutLMv3Processor.from_pretrained(config.model_name)
        self.model = InvoiceExtractor(config)
        
        if os.path.exists(model_path):
            # Assumes the model was saved as a state_dict
            model_weights_path = os.path.join(model_path, "pytorch_model.bin")
            self.model.load_state_dict(torch.load(model_weights_path))
        
        self.model.eval()
        self.ocr = DocumentOCR()
        self.label_to_field = {v: k for k, v in config.fields_to_extract.items()}

    def extract_from_image(self, image_path: str) -> Dict:
        if image_path.lower().endswith('.pdf'):
            image = convert_from_path(image_path)[0]
        else:
            image = Image.open(image_path).convert('RGB')
        
        ocr_data = self.ocr.extract_text_and_boxes(image)
        words, boxes = ocr_data['words'], ocr_data['boxes']
        
        width, height = image.size
        normalized_boxes = [
            [
                int(1000 * box[0] / width), int(1000 * box[1] / height),
                int(1000 * box[2] / width), int(1000 * box[3] / height)
            ] for box in boxes
        ]
        
        encoding = self.processor(
            image, words, boxes=normalized_boxes, truncation=True,
            padding="max_length", max_length=self.config.max_seq_length,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
        confidence_scores = torch.softmax(outputs.logits, dim=-1).squeeze()
        
        entities = self._extract_entities(words, predictions, confidence_scores)
        return {'extracted_fields': entities}

    def _extract_entities(self, words: List[str], predictions: List[int], confidences: torch.Tensor) -> Dict:
        entities = {}
        current_entity, current_text, current_confidence = None, [], []

        # Start from 1 to skip CLS token
        for i, pred_id in enumerate(predictions[1:]):
            if i >= len(words):
                break 

            pred_label = self.label_to_field.get(pred_id, 'O')
            word = words[i]
            confidence = confidences[i+1][pred_id].item()

            if pred_label.startswith('B-'):
                if current_entity:
                    entities[current_entity] = {
                        'text': ' '.join(current_text),
                        'confidence': np.mean(current_confidence)
                    }
                current_entity = pred_label[2:]
                current_text = [word]
                current_confidence = [confidence]
            elif pred_label.startswith('I-') and current_entity == pred_label[2:]:
                current_text.append(word)
                current_confidence.append(confidence)
            else:
                if current_entity:
                    entities[current_entity] = {
                        'text': ' '.join(current_text),
                        'confidence': np.mean(current_confidence)
                    }
                current_entity, current_text, current_confidence = None, [], []
        
        if current_entity:
            entities[current_entity] = {
                'text': ' '.join(current_text),
                'confidence': np.mean(current_confidence)
            }
        
        return entities