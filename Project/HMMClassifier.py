# HMMClassifier.py

import pickle
import numpy as np
from tqdm import tqdm
from hmmlearn import hmm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

###############################
#   Reuse your label parsing  #
###############################
def extract_label_and_fault_size(filename):
    if "Normal" in filename:
        return "Normal", None
    else:
        parts = filename.split("_")
        fault_type = parts[1]
        if "OR" in fault_type:
            label = "OR"
        elif "IR" in fault_type:
            label = "IR"
        elif "B" in fault_type:
            label = "B"
        else:
            label = fault_type
        try:
            fault_size = parts[2]
        except IndexError:
            fault_size = None
        return label, fault_size


def train_and_evaluate_hmm(
    preprocessed_path="preprocessed_data.pkl",
    n_states=3,
    pca_components=None,
    random_seed=None
):
    """
    1) Loads segmented signals from `preprocessed_data.pkl`
    2) Builds sequences (DE+FE columns)
    3) Splits into train/test by fault size
    4) (Optional) Applies PCA per time step
    5) Trains a GaussianHMM for each class
    6) Evaluates classification

    Returns (report, cm).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # 1) Load the preprocessed data
    with open(preprocessed_path, "rb") as f:
        segmented_signals = pickle.load(f)

    # 2) Create sequences
    sequences = []
    seq_labels = []
    keys = []

    for filename, channels in segmented_signals.items():
        # We assume near-raw DE and FE
        if "DE" in channels and "FE" in channels:
            de_wins = channels["DE"]
            fe_wins = channels["FE"]
            n_wins = min(len(de_wins), len(fe_wins))
            for i in range(n_wins):
                # shape (window_size, 2)
                seq = np.column_stack((de_wins[i], fe_wins[i]))
                label, _ = extract_label_and_fault_size(filename)

                sequences.append(seq)
                seq_labels.append(label)
                keys.append((filename, "merged", i))
        else:
            # If only one channel or none, skip
            pass

    # 3) Train/test split
    X_train, y_train, X_test, y_test = [], [], [], []
    for idx, key in enumerate(keys):
        filename, _, _ = key
        label, fault_size = extract_label_and_fault_size(filename)
        seq = sequences[idx]

        if label == "Normal":
            # random ~2:1 split
            if np.random.rand() < 0.67:
                X_train.append(seq)
                y_train.append(label)
            else:
                X_test.append(seq)
                y_test.append(label)
        else:
            # fault_size '7'/'14' → train, '21' → test
            if fault_size in {"7", "14"}:
                X_train.append(seq)
                y_train.append(label)
            elif fault_size == "21":
                X_test.append(seq)
                y_test.append(label)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # 4) PCA dimension reduction (time-step dimension)
    if pca_components is not None:
        all_train_timesteps = np.concatenate(X_train, axis=0)  # shape (sum_of_lengths, 2)
        pca_model = PCA(n_components=pca_components)
        pca_model.fit(all_train_timesteps)

        X_train = [pca_model.transform(seq) for seq in X_train]
        X_test = [pca_model.transform(seq) for seq in X_test]

    # 5) Train HMM per class
    classes = ["B", "IR", "Normal", "OR"]
    class_models = {}

    for cls in tqdm(classes, desc="Training HMM models"):
        # gather training seqs
        cls_seqs = [X_train[i] for i in range(len(X_train)) if y_train[i] == cls]
        if not cls_seqs:
            continue

        # Concat and store lengths
        lengths = [len(s) for s in cls_seqs]
        X_concat = np.concatenate(cls_seqs, axis=0)

        model = hmm.GaussianHMM(n_components=n_states, covariance_type='full',
                                n_iter=100, random_state=random_seed)
        model.fit(X_concat, lengths=lengths)
        class_models[cls] = model

    # 6) Prediction
    y_pred = []
    for seq in X_test:
        log_lik = {}
        for cls, model in class_models.items():
            try:
                score = model.score(seq)
            except:
                score = -1e10
            log_lik[cls] = score

        best_cls = max(log_lik, key=log_lik.get)
        y_pred.append(best_cls)

    y_pred = np.array(y_pred)

    # 7) Evaluate
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    report = classification_report(y_test, y_pred, labels=classes, output_dict=True)

    print("\n=== HMM Classification Results ===")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, labels=classes))
    print(f"Accuracy: {report['accuracy']:.4f}") 
    print("Confusion Matrix:\n", cm)

    return report, cm


if __name__ == "__main__":
    # EXAMPLE usage:
    # We'll load "preprocessed_data.pkl" (the segmented signals)
    train_and_evaluate_hmm(
        preprocessed_path="preprocessed_data.pkl",
        n_states=4,
        pca_components=None,
        random_seed=42
    )
