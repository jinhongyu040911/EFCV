import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(labels, preds):
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='binary', zero_division=0),
        'recall': recall_score(labels, preds, average='binary', zero_division=0),
        'f1': f1_score(labels, preds, average='binary', zero_division=0)
    }
