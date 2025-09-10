# src/dataset.py
import logging
import json
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
from .config import ExtractionConfig
import albumentations as A
import pytesseract
import torch

logger = logging.getLogger(__name__)

class DocumentOCR:
    """OCR processor for extracting text and bounding boxes from images"""
    def __init__(self):
        self.tesseract_config = '--oem 3 --psm 6'

    def extract_text_and_boxes(self, image: Image.Image) -> Dict:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        ocr_data = pytesseract.image_to_data(
            img_cv, config=self.tesseract_config, output_type=pytesseract.Output.DICT
        )
        words, boxes = [], []
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) > 30:
                word = ocr_data['text'][i].strip()
                if word:
                    x, y, w, h = (
                        ocr_data['left'][i], ocr_data['top'][i],
                        ocr_data['width'][i], ocr_data['height'][i]
                    )
                    words.append(word)
                    boxes.append([x, y, x + w, y + h])
        return {'words': words, 'boxes': boxes}

class DocumentAugmentation:
    """Data augmentation for document images"""
    def __init__(self):
        self.transform = A.Compose([
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
            ], p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def augment(self, image: np.ndarray, bboxes: List, labels: List) -> Tuple:
        try:
            augmented = self.transform(image=image, bboxes=bboxes, labels=labels)
            return augmented['image'], augmented['bboxes'], augmented['labels']
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}")
            return image, bboxes, labels

class InvoiceDataset(Dataset):
    """Dataset class for invoice documents"""
    def __init__(self, data_path: str, processor, config: ExtractionConfig, is_training: bool = True):
        self.data_path = Path(data_path)
        self.processor = processor
        self.config = config
        self.is_training = is_training
        self.ocr = DocumentOCR()
        self.augmentation = DocumentAugmentation()
        self.annotations = self._load_annotations()
        if len(self.annotations) == 0:
            raise ValueError(f"No data found in {data_path}. Make sure images or JSONs exist!")

    def _load_annotations(self) -> List[Dict]:
        annotations = []
        json_files = list(self.data_path.glob("*.json"))
        if json_files:
            for file_path in json_files:
                with open(file_path, 'r') as f:
                    annotations.append(json.load(f))
        else:
            for img_path in self.data_path.glob("*.*"):
                if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    annotations.append({"image_path": img_path.name, "entities": []})
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = self.data_path / annotation['image_path']
        image = Image.open(image_path).convert('RGB')

        ocr_data = self.ocr.extract_text_and_boxes(image)
        words, boxes = ocr_data['words'], ocr_data['boxes']

        labels = self._create_labels(words, boxes, annotation.get('entities', []))

        if self.is_training and np.random.random() < 0.7:
            image_np = np.array(image)
            image_np, boxes, labels = self.augmentation.augment(image_np, boxes, labels)
            image = Image.fromarray(image_np)

        # Ensure words, boxes, and labels have the same length
        min_len = min(len(words), len(boxes), len(labels))
        words = words[:min_len]
        boxes = boxes[:min_len]
        labels = labels[:min_len]

        width, height = image.size
        normalized_boxes = [
            [
                int(1000 * box[0] / width), int(1000 * box[1] / height),
                int(1000 * box[2] / width), int(1000 * box[3] / height)
            ] for box in boxes
        ]

        encoding = self.processor(
            image, words, boxes=normalized_boxes, word_labels=labels,
            truncation=True, padding="max_length",
            max_length=self.config.max_seq_length, return_tensors="pt"
        )

        # Convert labels to LongTensor for PyTorch
        encoding['labels'] = encoding['labels'].long()
        return {key: val.squeeze() for key, val in encoding.items()}

    def _create_labels(self, words: List[str], boxes: List, entities: List[Dict]) -> List[int]:
        labels = [self.config.fields_to_extract['O']] * len(words)
        for entity in entities:
            entity_type = entity['label']
            entity_text = entity['text'].lower()

            matched_indices = [
                i for i, word in enumerate(words)
                if word.lower() in entity_text.split()
            ]

            for i, word_idx in enumerate(matched_indices):
                if i == 0:
                    labels[word_idx] = self.config.fields_to_extract.get(f'B-{entity_type}', 0)
                else:
                    labels[word_idx] = self.config.fields_to_extract.get(f'I-{entity_type}', 0)
        return labels
