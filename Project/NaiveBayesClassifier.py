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
    Extracts the fault label and fault size from a given filename.

    This function parses a filename to determine the type of fault (e.g., "IR", "OR", "B", or "Normal") 
    and the fault size if applicable. If the filename indicates a "Normal" condition, the fault size 
    is returned as None.

    Args:
        filename (str): The filename string to be parsed. Expected format includes fault type 
                        and optionally fault size, e.g., '1730_IR_7_DE12' or '1730_Normal'.

    Returns:
        tuple: A tuple containing:
            - label (str): The fault label ("IR", "OR", "B", or "Normal").
            - fault_size (str or None): The fault size if present, otherwise None.
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
    Splits the feature set `X` into training and testing datasets based on the fault size and label.
    The splitting logic is as follows:
      - For samples labeled as "Normal", a random split is performed with a default ratio of 67% for training and 33% for testing.
      - For samples with fault sizes:
        - Fault sizes 7 or 14 are assigned to the training set.
        - Fault size 21 is assigned to the testing set.

    Args:
        X (list or np.ndarray): The feature set to be split.
        keys (list of tuples): A list of keys where each key is a tuple containing metadata about the sample.
        normal_split_ratio (float, optional): The ratio for splitting "Normal" samples into training and testing sets. Defaults to 0.67.
        seed (int, optional): The random seed for reproducibility of the "Normal" sample split. Defaults to 41.

    Returns:
        tuple: A tuple containing four numpy arrays:
            - X_train (np.ndarray): Features for the training set.
            - y_train (np.ndarray): Labels for the training set.
            - X_test (np.ndarray): Features for the testing set.
            - y_test (np.ndarray): Labels for the testing set.
    """
    random.seed(seed)  # optional
    X_train, y_train, X_test, y_test = [], [], [], []

    for i, key in enumerate(keys):
        filename, _, _ = key
        label, fault_size = extract_label_and_fault_size(filename)

        if label == "Normal": # Randomly split normal samples
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
    Flattens and normalizes a dictionary of features.

    Args:
        features_dict (dict): A dictionary where the keys are filenames and the values 
                              are lists of dictionaries containing feature-value pairs 
                              for each window. 
                              Format: {filename: [ {feat1: val, feat2: val, ...}, ... ]}
    Returns:
        tuple: A tuple containing:
            - X_normalized (numpy.ndarray): A 2D array of shape (N, d) where N is the 
              total number of feature windows and d is the number of features. The 
              features are normalized to have zero mean and unit variance.
            - keys (list): A list of tuples in the format (filename, "merged", window_idx), 
              where each tuple corresponds to a row in X_normalized.
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

    scaler = StandardScaler() # Normalize features to zero mean and unit variance
    X_normalized = scaler.fit_transform(data)
    return X_normalized, keys


###############################
#   The main NB training fn   #
###############################
def train_and_evaluate_classifier_from_features(features_path):
    """
    Loads feature data from a file, processes it, trains a Naive Bayes classifier, 
    and evaluates its performance using classification metrics.
    
    Args:
        features_path (str): Path to the file containing the serialized features dictionary.
        
    Returns:
        tuple: A tuple containing:
            - report (dict): A dictionary containing the classification report with metrics.
            - cm (numpy.ndarray): The confusion matrix of the classifier's predictions.
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
    # Print out classification results 
    print("\n=== Naive Bayes Classification Results ===")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print(f"Accuracy: {report['accuracy']:.4f}") 
    print("Confusion Matrix:\n", cm)

    return report, cm

if __name__ == "__main__":
    # Example usage:
    # Suppose you have "features_time.pkl" from `extract_features.py`
    train_and_evaluate_classifier_from_features("features_time.pkl")
