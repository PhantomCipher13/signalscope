import numpy as np

def compute_ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """
    Computes Expected Calibration Error using equal-width bins.
    
    ece = sum_m(|acc_m - conf_m| * n_m / N)
    
    Args:
        labels: 1D array of {0, 1}
        probs: 1D array in [0, 1]
        n_bins: number of bins
        
    Returns:
        float in [0, 1]
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    
    if len(labels) == 0 or len(probs) == 0:
        raise ValueError("Inputs cannot be empty")
        
    if len(labels) != len(probs):
        raise ValueError("Inputs must have the same length")
        
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(probs, bins, right=True)
    
    ece = 0.0
    for m in range(1, n_bins + 1):
        mask = (bin_indices == m)
        if m == 1:
            mask = mask | (probs == 0.0) # Include 0.0 in the first bin
            
        if np.any(mask):
            acc_m = np.mean(labels[mask])
            conf_m = np.mean(probs[mask])
            n_m = np.sum(mask)
            ece += np.abs(acc_m - conf_m) * (n_m / len(labels))
            
    return float(ece)

def compute_brier(labels: np.ndarray, probs: np.ndarray) -> float:
    """
    Computes Mean squared error of probability predictions.
    
    brier = mean((prob - label)^2)
    
    Args:
        labels: 1D array of {0, 1}
        probs: 1D array in [0, 1]
        
    Returns:
        float in [0, 1]
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    
    if len(labels) == 0 or len(probs) == 0:
        raise ValueError("Inputs cannot be empty")
        
    return float(np.mean((probs - labels) ** 2))

def reliability_diagram_data(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> dict:
    """
    Computes data for reliability diagram visualisation.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(probs, bins, right=True)
    
    bin_centers = []
    mean_confidence = []
    fraction_positive = []
    bin_counts = []
    
    for m in range(1, n_bins + 1):
        mask = (bin_indices == m)
        if m == 1:
            mask = mask | (probs == 0.0)
            
        bin_center = (bins[m-1] + bins[m]) / 2.0
        bin_centers.append(bin_center)
        
        if np.any(mask):
            mean_confidence.append(np.mean(probs[mask]))
            fraction_positive.append(np.mean(labels[mask]))
            bin_counts.append(np.sum(mask))
        else:
            mean_confidence.append(np.nan)
            fraction_positive.append(np.nan)
            bin_counts.append(0)
            
    return {
        "bin_centers": np.array(bin_centers),
        "mean_confidence": np.array(mean_confidence),
        "fraction_positive": np.array(fraction_positive),
        "bin_counts": np.array(bin_counts)
    }
