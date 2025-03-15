from BearingData import (
    files_1730RPM,
    files_1750RPM,
    files_1772RPM,
    files_1797RPM
)
from LoadSignals import load_signals 
from Preprocess import downsample, segment_signals
from ExtractFeatures import time_domain_features, frequency_domain_features, merge_channel_features
from NaiveBayesClassifier import normalize_features, train_and_evaluate_classifier
import numpy as np

def run_pipeline():
    """
    Runs the entire pipeline once:
      - Loads signals from all RPM dictionaries
      - Combines, downsamples, segments, and extracts features
      - Merges DE and FE features
      - Normalizes the merged features and trains/evaluates the Naïve Bayes classifier
    Returns the classification report (as a dictionary) and the confusion matrix.
    """
    # Load signals for all RPMs
    signals_1730RPM = load_signals(files_1730RPM)
    signals_1750RPM = load_signals(files_1750RPM)
    signals_1772RPM = load_signals(files_1772RPM)
    signals_1797RPM = load_signals(files_1797RPM)
    
    # Combine signals from all four RPM dictionaries into a single dictionary
    all_signals = {}
    all_signals.update(signals_1730RPM)
    all_signals.update(signals_1750RPM)
    all_signals.update(signals_1772RPM)
    all_signals.update(signals_1797RPM)
    
    # Preprocess the combined signals: downsample and segment.
    ds_signals = downsample(all_signals)
    seg_signals = segment_signals(ds_signals)
    
    # Extract time-domain (or frequency-domain) features. Here we use time-domain.
    td_features = time_domain_features(seg_signals)
    # Merge DE and FE features so that each window is represented by one combined feature vector.
    merged_features = merge_channel_features(td_features)
    
    # Normalize the extracted features and flatten the structure to get a 2D feature matrix.
    X_norm, feature_keys = normalize_features(merged_features)
    
    # Train and evaluate the Naïve Bayes classifier on the combined dataset.
    # Splitting: faulty data with sizes '7'/'14' for training, '21' for testing; Normal data split 2:1.
    report, cm = train_and_evaluate_classifier(X_norm, feature_keys)
    return report, cm

def average_reports(reports):
    """
    Averages a list of classification report dictionaries.
    
    Parameters:
        reports (list): List of classification report dictionaries (one per run).
    
    Returns:
        avg_report (dict): Dictionary with the averaged metrics.
    """
    avg_report = {}
    n = len(reports)
    for rep in reports:
        for key, metrics in rep.items():
            # Initialize based on type
            if key not in avg_report:
                if isinstance(metrics, dict):
                    avg_report[key] = {}
                else:
                    avg_report[key] = 0.0
            # Add the metrics based on their type
            if isinstance(metrics, dict):
                for m_key, m_val in metrics.items():
                    avg_report[key][m_key] = avg_report[key].get(m_key, 0) + m_val
            else:
                avg_report[key] += metrics
    # Divide each accumulated value by n
    for key, metrics in avg_report.items():
        if isinstance(metrics, dict):
            for m_key in metrics:
                avg_report[key][m_key] = round(metrics[m_key] / n, 2)
        else:
            avg_report[key] = round(metrics / n, 2)
    return avg_report

def main():
    n_runs = 30
    reports = []
    
    for i in range(n_runs):
        print(f"\n--- Run {i+1} ---")
        report, cm = run_pipeline()
        reports.append(report)
    
    avg_report = average_reports(reports)
    print("\n=== Average Classification Report over 30 runs ===")
    # For pretty printing, you might iterate over the avg_report dictionary.
    for class_label, metrics in avg_report.items():
        print(f"{class_label}: {metrics}")

if __name__ == "__main__":
    main()
