# run.py
# This script runs the entire pipeline for preprocessing, feature extraction, and classification.
# It assumes that the necessary functions are defined in the respective modules.

import random
import os
import matplotlib.pyplot as plt
from pathlib import Path
from prepare_data import preprocess_and_save
from extract_features import extract_and_save
from HMMClassifier import train_and_evaluate_hmm 
from NaiveBayesClassifier import train_and_evaluate_classifier_from_features

def run(window_size): 
    # Set a random seed for reproducibility 
    seed = 40
    print(f"\n=== Running pipeline with window size = {window_size} ===")
    # Step 1: Preprocess data
    print("Step 1: Preprocessing data...")
    # This will create a file 'preprocessed_data.pkl' in the current directory
    preprocess_and_save(window_size=window_size, overlap=0.5, out_path="preprocessed_data.pkl")

    # Step 2: Extract features (time-domain in this case) for the Naive Bayes model.
    print("\nStep 2: Extracting time-domain features...")
    # This will create 'features_time.pkl'
    extract_and_save(preprocessed_path="preprocessed_data.pkl",
                    feature_type="time",
                    out_path="features_time.pkl",
                    fs=12000)

    # Step 3: Run the Naive Bayes classifier and capture NB accuracy
    print("\nStep 3: Running Naive Bayes classifier...")
    nb_report, nb_cm = train_and_evaluate_classifier_from_features("features_time.pkl")
    nb_accuracy = nb_report.get("accuracy", 0)
    print(f"NB Accuracy for window size {window_size}: {nb_accuracy:.4f}")

    # Step 4: Run the HMM classifier and capture HMM accuracy
    print("\nStep 4: Running HMM classifier...")
    report, _ = train_and_evaluate_hmm(preprocessed_path="preprocessed_data.pkl",
                                    n_states=3,
                                    pca_components=None,
                                    random_seed=seed)
    hmm_accuracy = report.get("accuracy", 0)
    print(f"HMM Accuracy for window size {window_size}: {hmm_accuracy:.4f}")

    return hmm_accuracy, nb_accuracy

if __name__ == "__main__": 
    window_sizes = [500, 1000, 2000, 3000, 4000, 6000, 8000, 12000]
    hmm_accuracies = []
    nb_accuracies = []

    # Run pipeline for each window size and collect accuracies
    for ws in window_sizes:
        hmm_acc, nb_acc = run(ws)
        hmm_accuracies.append(hmm_acc)
        nb_accuracies.append(nb_acc)

    # Graph 1: Both HMM and NB Accuracy vs Window Size
    plt.figure(figsize=(8,6))
    plt.plot(window_sizes, hmm_accuracies, marker='o', linestyle='-', color='blue', label="HMM Accuracy")
    plt.plot(window_sizes, nb_accuracies, marker='o', linestyle='-', color='red', label="NB Accuracy")
    plt.xlabel("Window Size")
    plt.ylabel("Accuracy")
    plt.title("HMM and NB Accuracy vs Window Size")
    plt.legend()
    plt.grid(True)
    downloads_path1 = Path.home() / "Downloads" / "HMM_and_NB_accuracy_vs_window_size.png"
    plt.savefig(downloads_path1)
    print(f"\nGraph saved to: {downloads_path1}")
    plt.show()
