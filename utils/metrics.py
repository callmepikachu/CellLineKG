"""
Utility functions for CellLineKG project.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score

def compute_auc_roc(true_labels, predictions):
    """
    Compute AUC-ROC score.
    
    Args:
        true_labels (array-like): True binary labels.
        predictions (array-like): Predicted probabilities.
        
    Returns:
        float: AUC-ROC score.
    """
    return roc_auc_score(true_labels, predictions)

def compute_precision_at_k(true_labels, predictions, k=10):
    """
    Compute Precision@K.
    
    Args:
        true_labels (array-like): True binary labels.
        predictions (array-like): Predicted probabilities.
        k (int): Top K predictions to consider.
        
    Returns:
        float: Precision@K score.
    """
    # Get indices of top K predictions
    top_k_indices = np.argsort(predictions)[::-1][:k]
    # Get true labels for top K predictions
    top_k_labels = np.array(true_labels)[top_k_indices]
    # Compute precision
    return precision_score(top_k_labels, [1]*len(top_k_labels), zero_division=0)

def compute_recall_at_k(true_labels, predictions, k=10):
    """
    Compute Recall@K.
    
    Args:
        true_labels (array-like): True binary labels.
        predictions (array-like): Predicted probabilities.
        k (int): Top K predictions to consider.
        
    Returns:
        float: Recall@K score.
    """
    # Get indices of top K predictions
    top_k_indices = np.argsort(predictions)[::-1][:k]
    # Get true labels for top K predictions
    top_k_labels = np.array(true_labels)[top_k_indices]
    # Compute recall
    return recall_score(true_labels, [1 if i in top_k_indices else 0 for i in range(len(predictions))], zero_division=0)

def save_model(model, path):
    """
    Save model to disk.
    
    Args:
        model (nn.Module): Model to save.
        path (str): Path to save model.
    """
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """
    Load model from disk.
    
    Args:
        model (nn.Module): Model to load weights into.
        path (str): Path to load model from.
    """
    model.load_state_dict(torch.load(path))
    return model