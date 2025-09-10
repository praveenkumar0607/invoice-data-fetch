# src/evaluation.py
import numpy as np
from typing import Dict, List
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Evaluation metrics for extraction accuracy"""
    @staticmethod
    def compute_field_accuracy(predictions: Dict, ground_truth: Dict) -> Dict:
        metrics = {}
        all_fields = set(predictions.keys()) | set(ground_truth.keys())
        
        for field in all_fields:
            pred_text = predictions.get(field, {}).get('text', '').lower().strip()
            true_text = ground_truth.get(field, '').lower().strip()
            
            exact_match = pred_text == true_text
            
            pred_words = set(pred_text.split())
            true_words = set(true_text.split())
            iou = 0.0
            if len(pred_words | true_words) > 0:
                iou = len(pred_words & true_words) / len(pred_words | true_words)

            metrics[field] = {
                'exact_match': exact_match,
                'iou': iou,
                'confidence': predictions.get(field, {}).get('confidence', 0.0)
            }
        return metrics
    
    # You can add the other evaluation methods here...