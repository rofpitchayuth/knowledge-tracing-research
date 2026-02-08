"""
Evaluation metrics for Knowledge Tracing models.

Provides common metrics used in KT research including AUC, accuracy,
RMSE, log-likelihood, and calibration measures.
"""

from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_fscore_support,
    mean_squared_error, log_loss, confusion_matrix
)


def compute_all_metrics(predictions: np.ndarray, targets: np.ndarray, 
                       threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute all standard metrics for KT evaluation.
    
    Args:
        predictions: Predicted probabilities (0-1)
        targets: True labels (0 or 1)
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary of metric names and values
    """
    # Convert to numpy arrays
    predictions = np.array(predictions)
    targets = np.array(targets).astype(int)
    
    # Binary predictions
    binary_preds = (predictions >= threshold).astype(int)
    
    # Basic metrics
    metrics = {
        'auc': compute_auc(predictions, targets),
        'accuracy': float(accuracy_score(targets, binary_preds)),
        'rmse': compute_rmse(predictions, targets),
        'log_loss': compute_log_loss(predictions, targets),
    }
    
    # Precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, binary_preds, average='binary', zero_division=0
    )
    metrics['precision'] = float(precision)
    metrics['recall'] = float(recall)
    metrics['f1'] = float(f1)
    
    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(targets, binary_preds).ravel()
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    
    # Calibration
    metrics['ece'] = compute_expected_calibration_error(predictions, targets)
    
    return metrics


def compute_auc(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Area Under ROC Curve."""
    try:
        return float(roc_auc_score(targets, predictions))
    except ValueError:
        # Handle case where only one class is present
        return 0.5


def compute_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(targets, predictions)))


def compute_log_loss(predictions: np.ndarray, targets: np.ndarray, 
                     eps: float = 1e-10) -> float:
    """
    Compute log loss (cross-entropy).
    
    Args:
        predictions: Predicted probabilities
        targets: True labels
        eps: Small constant to avoid log(0)
    """
    # Clip predictions to avoid log(0)
    predictions = np.clip(predictions, eps, 1 - eps)
    
    try:
        return float(log_loss(targets, predictions, labels=[0, 1]))
    except ValueError:
        # Manual calculation if sklearn fails
        return float(-np.mean(
            targets * np.log(predictions) + 
            (1 - targets) * np.log(1 - predictions)
        ))


def compute_expected_calibration_error(predictions: np.ndarray, targets: np.ndarray,
                                      num_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures how well the predicted probabilities match the actual outcomes.
    Lower is better (0 = perfect calibration).
    
    Args:
        predictions: Predicted probabilities
        targets: True labels
        num_bins: Number of bins for calibration
        
    Returns:
        ECE value
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(targets[in_bin])
            avg_confidence_in_bin = np.mean(predictions[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


def compute_likelihood(predictions: np.ndarray, targets: np.ndarray,
                      eps: float = 1e-10) -> float:
    """
    Compute likelihood of data given predictions.
    
    Args:
        predictions: Predicted probabilities
        targets: True labels (0 or 1)
        eps: Small constant to avoid log(0)
        
    Returns:
        Log-likelihood value
    """
    predictions = np.clip(predictions, eps, 1 - eps)
    
    log_likelihood = np.sum(
        targets * np.log(predictions) + 
        (1 - targets) * np.log(1 - predictions)
    )
    
    return float(log_likelihood)


def compute_brier_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Brier score (mean squared error for probabilities).
    
    Lower is better (0 = perfect predictions).
    """
    return float(np.mean((predictions - targets) ** 2))


def print_metric_summary(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """
    Pretty print metric summary.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the summary
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    # Group metrics
    accuracy_metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1']
    error_metrics = ['rmse', 'log_loss', 'ece']
    
    print("\nAccuracy Metrics:")
    for key in accuracy_metrics:
        if key in metrics:
            print(f"  {key.upper():12s}: {metrics[key]:.4f}")
    
    print("\nError Metrics:")
    for key in error_metrics:
        if key in metrics:
            print(f"  {key.upper():12s}: {metrics[key]:.4f}")
    
    if 'true_positives' in metrics:
        print("\nConfusion Matrix:")
        print(f"  True Positives:  {metrics['true_positives']:6d}")
        print(f"  True Negatives:  {metrics['true_negatives']:6d}")
        print(f"  False Positives: {metrics['false_positives']:6d}")
        print(f"  False Negatives: {metrics['false_negatives']:6d}")
    
    print(f"{'='*60}\n")
