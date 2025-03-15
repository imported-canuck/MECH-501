# NaiveBayesClassifier.py

import random
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def extract_label_and_fault_size(filename):
    """
    Given a filename in the format 'RPM_Fault_Size_LocationSamplerate'
    (e.g., '1730_IR_7_DE12') or 'RPM_Normal', extract the class label and, for faulty files,
    the fault size.
    
    Returns:
        label (str): One of 'Normal', 'IR', 'OR', or 'B'
        fault_size (str or None): e.g., '7', '14', '21' for faulty files; None for Normal.
    """
    if "Normal" in filename:
        return "Normal", None
    else:
        parts = filename.split("_")
        # parts[0] is the RPM, parts[1] is the fault type, parts[2] is the fault size.
        fault_type = parts[1]
        # Map any outer race type to "OR"
        if "OR" in fault_type:
            label = "OR"
        elif "IR" in fault_type:
            label = "IR"
        elif "B" in fault_type:
            label = "B"
        else:
            label = fault_type  # fallback (shouldn't happen)
        try:
            fault_size = parts[2]
        except IndexError:
            fault_size = None
        return label, fault_size

def split_features_by_fault_size(X, keys, normal_split_ratio=0.67, seed=41):
    """ 
    Parameters:
        X (np.array): Normalized feature matrix of shape (n_samples, n_features)
        keys (list): List of tuples (filename, channel, window_index) for each row in X.
        normal_split_ratio (float): Fraction of Normal samples to assign to training.
        seed (int): Random seed for reproducibility.
    
    Returns:
        X_train, y_train, X_test, y_test: Arrays with features and corresponding labels.
    
    Splits the normalized feature matrix X (with corresponding keys) into training and testing sets.
    
    For non-normal (faulty) samples:
        - Files with fault size '7' or '14' are assigned to training.
        - Files with fault size '21' are assigned to testing.
    For Normal samples, a random split is performed with the given ratio (normal_split_ratio for training).
   
    """
    # random.seed(seed)
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    for i, key in enumerate(keys):
        filename, channel, window_idx = key
        label, fault_size = extract_label_and_fault_size(filename)
        
        if label == "Normal":
            # Randomly assign normal samples based on the specified ratio.
            if random.random() < normal_split_ratio:
                X_train.append(X[i])
                y_train.append(label)
            else:
                X_test.append(X[i])
                y_test.append(label)
        else:
            # Faulty data: if fault size is '7' or '14', assign to training; if '21', assign to test.
            if fault_size in {"7", "14"}:
                X_train.append(X[i])
                y_train.append(label)
            elif fault_size == "21":
                X_test.append(X[i])
                y_test.append(label)
    
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def normalize_features(features_dict):
    """
    Parameters: 
        features_dict (dict): Dictionary with one of the following structures:
            Case 1 (separate channels):
            { filename: { 
                  'DE': [ {feat1: value, feat2: value, ...}, ... ],
                  'FE': [ {feat1: value, feat2: value, ...}, ... ]
              },
              ... }
            Case 2 (merged channels):
            { filename: [ {feat1: value, feat2: value, ...}, ... ],
              ... }

    Returns:
        X_normalized (np.array): Normalized feature matrix of shape (n_samples, n_features).
        keys (list): A list of tuples (filename, channel, window_index) corresponding to each sample.
    
    Places feature dictionaries into a matrix, yielding a matrix contianing the 
    time/frequency domain features of each data point per row. keys provides the
    mapping to the files each row of X_normalized belongs to.
    """
    data = []
    keys = []
    for filename, value in features_dict.items():
        if isinstance(value, dict):
            # Process separate channels
            for channel, feats_list in value.items():
                for idx, feat_dict in enumerate(feats_list):
                    ordered_feats = [feat_dict[k] for k in sorted(feat_dict.keys())]
                    data.append(ordered_feats)
                    keys.append((filename, channel, idx))
        elif isinstance(value, list):
            # Process merged features
            for idx, feat_dict in enumerate(value):
                ordered_feats = [feat_dict[k] for k in sorted(feat_dict.keys())]
                data.append(ordered_feats)
                keys.append((filename, "merged", idx))
    data = np.array(data)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(data)
    return X_normalized, keys

def train_and_evaluate_classifier(X, keys):
    """
    Given a normalized feature matrix X and corresponding keys, splits the data into training and testing sets,
    trains a Gaussian Naïve Bayes classifier, and prints evaluation metrics.
    
    Parameters:
        X (np.array): Normalized feature matrix of shape (n_samples, n_features)
        keys (list): List of tuples (filename, channel, window_index) for each row in X.
        
    Returns:
        report (dict): Classification report dictionary.
        cm (np.array): Confusion matrix.
    """
    # Split data based on fault size and Normal random split
    X_train, y_train, X_test, y_test = split_features_by_fault_size(X, keys)
    
    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])
    
    # Train Gaussian Naïve Bayes classifier
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = clf.predict(X_test)
    
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(cm)
    
    return report, cm

if __name__ == "__main__":
    # For testing purposes, create dummy data.
    X_dummy = np.random.randn(10, 7)  # 10 samples, 7 features each
    # Create dummy keys with different fault sizes:
    dummy_keys = [
        ("1730_IR_7_DE12", "DE", 0),
        ("1730_IR_14_DE12", "DE", 1),
        ("1730_IR_21_DE12", "DE", 2),
        ("1730_B_7_DE12", "DE", 0),
        ("1730_B_14_DE12", "DE", 1),
        ("1730_B_21_DE12", "DE", 2),
        ("1730_Normal", "DE", 0),
        ("1730_Normal", "DE", 1),
        ("1730_Normal", "DE", 2),
        ("1730_Normal", "DE", 3),
    ]
    train_and_evaluate_classifier(X_dummy, dummy_keys)
