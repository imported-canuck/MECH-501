from BearingData import (
    files_1730RPM,
    files_1750RPM,
    files_1772RPM,
    files_1797RPM
)
from LoadSignals import load_signals 
from Preprocess import downsample, segment_signals
from ExtractFeatures import time_domain_features, frequency_domain_features
from NaiveBayesClassifier import normalize_features, train_and_evaluate_classifier
import numpy as np
import matplotlib.pyplot as plt

def merge_time_and_freq_features(td_features, fd_features):
    """
    Merges time-domain and frequency-domain features for each file on a window-by-window basis.
    
    Assumes that both td_features and fd_features are dictionaries with keys as filenames and values
    as lists of merged feature dictionaries (one per window). For each window, this function creates a
    new dictionary that contains all time-domain features (prefixed with "TD_") and all frequency-domain
    features (prefixed with "FD_").
    
    Parameters:
        td_features (dict): Merged time-domain feature dictionary:
            { filename: [ {feat1: value, ...}, {feat1: value, ...}, ... ], ... }
        fd_features (dict): Merged frequency-domain feature dictionary with the same structure.
    
    Returns:
        merged_features (dict): Dictionary with the structure:
            { filename: [ { "TD_<feat>": value, "FD_<feat>": value, ... }, ... ], ... }
    """
    merged_features = {}
    for filename in td_features:
        merged_features[filename] = []
        # Get the number of windows (take the minimum in case of a mismatch)
        n_windows = min(len(td_features[filename]), len(fd_features.get(filename, [])))
        for i in range(n_windows):
            merged_dict = {}
            # Add time-domain features with prefix "TD_"
            for k, v in td_features[filename][i].items():
                merged_dict["TD_" + k] = v
            # Add frequency-domain features with prefix "FD_"
            for k, v in fd_features[filename][i].items():
                merged_dict["FD_" + k] = v
            merged_features[filename].append(merged_dict)
    return merged_features


def run_pipeline_features(feature_set, window_size, overlap):
    """
    Runs the entire pipeline once for a given feature set, window size, and overlap.
    
    Parameters:
        feature_set (str): One of "time", "freq", or "both"
        window_size (int): The window size to use for segmentation.
        overlap (float): The fractional overlap between consecutive windows.
    
    Returns:
        report (dict): Classification report dictionary.
        cm (np.array): Confusion matrix.
    """
    # Load signals from all RPM dictionaries
    signals_1730RPM = load_signals(files_1730RPM)
    signals_1750RPM = load_signals(files_1750RPM)
    signals_1772RPM = load_signals(files_1772RPM)
    signals_1797RPM = load_signals(files_1797RPM)
    
    # Combine signals
    all_signals = {}
    all_signals.update(signals_1730RPM)
    all_signals.update(signals_1750RPM)
    all_signals.update(signals_1772RPM)
    all_signals.update(signals_1797RPM)
    
    # Preprocess: downsample and segment (using given window_size and overlap)
    ds_signals = downsample(all_signals)
    seg_signals = segment_signals(ds_signals, window_size, overlap)
    
    if feature_set == "time":
        features = time_domain_features(seg_signals)
    elif feature_set == "freq":
        features = frequency_domain_features(seg_signals, fs=12000)
    elif feature_set == "both":
        td_feats = time_domain_features(seg_signals)
        fd_feats = frequency_domain_features(seg_signals, fs=12000)
        features = merge_time_and_freq_features(td_feats, fd_feats)
    else:
        raise ValueError("Unknown feature set: choose from 'time', 'freq', or 'both'")
    
    X_norm, feature_keys = normalize_features(features)
    report, cm = train_and_evaluate_classifier(X_norm, feature_keys)
    return report, cm

def main():
    window_sizes = list(range(500, 12000 + 1, 500))
    overlap_values = [0.25] ### EDITING
    feature_sets = ["time"]
    # Dictionary to store results: results[feature_set][overlap][window_size] = accuracy
    results = {fs: {ov: {} for ov in overlap_values} for fs in feature_sets}
    
    for fs in feature_sets:
        for ov in overlap_values:
            print(f"\n=== Running experiments for feature set: {fs}, overlap: {ov} ===")
            for ws in window_sizes:
                print(f"\nTesting window size: {ws}")
                report, cm = run_pipeline_features(fs, ws, ov)
                accuracy = report.get("accuracy", 0)
                results[fs][ov][ws] = round(accuracy, 4)
                print(f"Feature set: {fs}, Overlap: {ov}, Window size: {ws}, Accuracy: {results[fs][ov][ws]:.4f}")
    
    # Plot 12 graphs: for each feature set and each overlap
    for fs in feature_sets:
        for ov in overlap_values:
            ws_list = sorted(results[fs][ov].keys())
            acc_list = [results[fs][ov][w] for w in ws_list]
            plt.figure()
            plt.plot(ws_list, acc_list, marker='o')
            plt.xlabel("Window Size")
            plt.ylabel("Accuracy")
            plt.title(f"Accuracy vs Window Size\nFeature set: {fs.capitalize()}, Overlap: {ov}")
            plt.grid(True)
            plt.ylim(0, 1)
            plt.xticks(ws_list, rotation=45)
            plt.tight_layout()
            save_path = f"C:/Users/cagri/Downloads/accuracy_{fs}_{ov}.png"
            plt.savefig(save_path)
            plt.close()
            print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    main()
