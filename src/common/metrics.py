import numpy as np
import time
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix

def calculate_ece(confidences, predictions, true_labels, n_bins=15):
    """
    Calculates Expected Calibration Error with 15 equal-width confidence bins.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = np.where((confidences > bin_lower) & (confidences <= bin_upper))[0]
        
        if len(in_bin) > 0:
            bin_acc = np.mean(predictions[in_bin] == true_labels[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            bin_weight = len(in_bin) / len(confidences)
            ece += bin_weight * np.abs(bin_acc - bin_conf)
            
    return float(ece)

def calculate_multiclass_brier_score(probabilities, true_labels, n_classes=4):
    """
    Calculates the multi-class Brier score.
    """
    # Convert true labels to one-hot encoding
    one_hot_labels = np.eye(n_classes)[true_labels]
    brier_score = np.mean(np.sum((probabilities - one_hot_labels) ** 2, axis=1))
    return float(brier_score)

def evaluate_classification(probabilities, true_labels):
    """
    Generates all primary and secondary metrics.
    probabilities: Array of shape (N, 4)
    true_labels: Array of shape (N,)
    """
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    acc = accuracy_score(true_labels, predictions)
    
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(true_labels, predictions, labels=[0,1,2,3])
    cm = confusion_matrix(true_labels, predictions, labels=[0,1,2,3])
    
    ece = calculate_ece(confidences, predictions, true_labels, n_bins=15)
    brier = calculate_multiclass_brier_score(probabilities, true_labels, n_classes=4)
    
    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
        "per_class_f1": f1_per_class.tolist(),
        "confusion_matrix": cm.tolist(),
        "ece": ece,
        "brier_score": brier
    }

def measure_latency(model, sample_input, device):
    """
    Measures batch-one inference latency (100 warmup, 1000 timed inferences).
    """
    # Exclude data loading by keeping the sample on device
    sample_input = sample_input.to(device)
    
    # Warmup
    for _ in range(100):
        _ = model(sample_input)
        
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        _ = model(sample_input)
        end = time.perf_counter()
        times.append(end - start)
        
    return {
        "mean_latency": np.mean(times),
        "p95_latency": np.percentile(times, 95)
    }