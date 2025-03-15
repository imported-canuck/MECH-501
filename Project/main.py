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

def main():
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
    
    # Preprocess the combined signals:
    # 1. Downsample (e.g., 48 kHz to 12 kHz)
    ds_signals = downsample(all_signals)
    # 2. Segment the downsampled signals into overlapping windows
    seg_signals = segment_signals(ds_signals)
    
    # Extract time-domain (or frequency-domain) features
    td_features = time_domain_features(seg_signals)
    # Merge DE and FE features so that each window is represented by one combined feature vector
    merged_features = merge_channel_features(td_features)
    
    # Normalize the extracted features and flatten the structure to get a 2D feature matrix.
    X_norm, feature_keys = normalize_features(merged_features)
    
    # Train and evaluate the Naïve Bayes classifier on the combined dataset.
    # Split as follows:
    #   - For faulty bearings, use fault sizes '7' and '14' for training and '21' for testing.
    #   - For Normal data, perform a random 2:1 split between train and test.
    train_and_evaluate_classifier(X_norm, feature_keys)

if __name__ == "__main__":
    main()