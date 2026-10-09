import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Your code here
    data=np.array(data)
    mean=np.mean(data)
    median = np.median(data)
    values,counts=np.unique(data,return_counts=True)
    mode=values[np.argmax(counts)]
    variance=np.var(data)
    standard_deviation = np.std(data)
    p25, p50, p75 = np.percentile(data, [25, 50, 75])
    iqr = p75 - p25
    return {
        "mean": float(mean),
        "median": float(median),
        "mode": mode.item(),
        "variance": float(variance),
        "standard_deviation": float(standard_deviation),
        "25th_percentile": float(p25),
        "50th_percentile": float(p50),
        "75th_percentile": float(p75),
        "interquartile_range": float(iqr)
    }