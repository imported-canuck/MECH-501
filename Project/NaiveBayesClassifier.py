# NaiveBayesClassifier.py

import pickle
import numpy as np
import random
from tqdm import tqdm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


def extract_label_and_fault_size(filename):
    """
    E.g. '1730_IR_7_DE12' → ('IR', '7'), '1730_Normal' → ('Normal', None).
    """
    if "Normal" in filename:
        return "Normal", None
    else:
        parts = filename.split("_")
        fault_type = parts[1]
        # Map "OR", "IR", "B"
        if "OR" in fault_type:
            label = "OR"
        elif "IR" in fault_type:
            label = "IR"
        elif "B" in fault_type:
            label = "B"
        else:
            label = fault_type
        # Fault size if present
        try:
            fault_size = parts[2]
        except IndexError:
            fault_size = None
        return label, fault_size

def split_features_by_fault_size(X, keys, normal_split_ratio=0.67, seed=41):
    """
    Splits X into train/test:
      - Normal → random split (67% train, 33% test).
      - Fault sizes 7 or 14 → train, 21 → test.
    """
    random.seed(seed)  # optional
    X_train, y_train, X_test, y_test = [], [], [], []

    for i, key in enumerate(keys):
        filename, _, _ = key
        label, fault_size = extract_label_and_fault_size(filename)

        if label == "Normal":
            if random.random() < normal_split_ratio:
                X_train.append(X[i])
                y_train.append(label)
            else:
                X_test.append(X[i])
                y_test.append(label)
        else:
            # Fault sizes 7/14 → train, 21 → test
            if fault_size in {"7", "14"}:
                X_train.append(X[i])
                y_train.append(label)
            elif fault_size == "21":
                X_test.append(X[i])
                y_test.append(label)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def normalize_features(features_dict):
    """
    Flatten and normalize a features dictionary:
      {filename: [ {feat1: val, feat2: val, ...}, ... ] }

    Returns (X_normalized, keys):
      X_normalized: (N, d)
      keys: list of (filename, "merged", window_idx)
    """
    data = []
    keys = []
    for filename, feat_list in features_dict.items():
        for idx, feat_dict in enumerate(feat_list):
            # Sort features by key name so they're consistently ordered
            row = [feat_dict[k] for k in sorted(feat_dict.keys())]
            data.append(row)
            keys.append((filename, "merged", idx))
    data = np.array(data)

    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(data)
    return X_normalized, keys


###############################
#   The main NB training fn   #
###############################
def train_and_evaluate_classifier_from_features(features_path):
    """
    Loads the features dict from `features_path`,
    normalizes them, splits them, trains NB, prints metrics.
    """
    # 1) Load features
    with open(features_path, 'rb') as f:
        features_dict = pickle.load(f)

    # 2) Flatten + normalize
    X, all_keys = normalize_features(features_dict)

    # 3) Split into train/test
    X_train, y_train, X_test, y_test = split_features_by_fault_size(X, all_keys)

    print(f"[INFO] Train set size = {X_train.shape[0]}, Test set size = {X_test.shape[0]}")

    # 4) Train
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    # 5) Evaluate with progress bar
    y_pred = []
    for sample in tqdm(X_test, desc="Predicting NB"):
        # Predict for each sample individually to update progress bar
        y_pred.append(clf.predict(sample.reshape(1, -1))[0])
    y_pred = np.array(y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Naive Bayes Classification Results ===")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print(f"Accuracy: {report['accuracy']:.4f}") 
    print("Confusion Matrix:\n", cm)

    return report, cm


if __name__ == "__main__":
    # EXAMPLE USAGE:
    # Suppose you have "features_time.pkl" from `extract_features.py`
    train_and_evaluate_classifier_from_features("features_time.pkl")
