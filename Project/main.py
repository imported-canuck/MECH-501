# main.py

import random
import argparse
from NaiveBayesClassifier import train_and_evaluate_classifier_from_features
from HMMClassifier import train_and_evaluate_hmm

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run either Naive Bayes or HMM classification on preprocessed bearing data."
    )
    parser.add_argument(
        "--model",
        choices=["naive", "hmm"],
        required=True,
        help="Which classifier to run: 'naive' (Naive Bayes) or 'hmm' (Hidden Markov Model)."
    )
    parser.add_argument(
        "--features_path",
        default="features_time.pkl",
        help="Path to the features .pkl file (for Naive Bayes). Defaults to 'features_time.pkl'."
    )
    parser.add_argument(
        "--preprocessed_path",
        default="preprocessed_data.pkl",
        help="Path to the preprocessed signals .pkl file (for HMM). Defaults to 'preprocessed_data.pkl'."
    )
    parser.add_argument(
        "--n_states",
        type=int,
        default=3,
        help="Number of hidden states in the HMM (only used if model='hmm')."
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=None,
        help="If provided, reduce each time-step dimension with PCA. (Only for HMM.)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for splitting data & training (both NB and HMM)."
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.random_seed is None:
        args.random_seed = random.randint(0, 10**6)

    if args.model == "naive":
        # Run Naive Bayes classifier with the specified features file
        # The random seed for NB splits is set inside the script,
        # but we can override it easily by passing it into the split function if desired.
        print(f"[INFO] Running Naive Bayes with features: {args.features_path}")
        train_and_evaluate_classifier_from_features(args.features_path)

    elif args.model == "hmm":
        # Run HMM classifier with the specified preprocessed data
        print(f"[INFO] Running HMM with preprocessed data: {args.preprocessed_path}")
        train_and_evaluate_hmm(
            preprocessed_path=args.preprocessed_path,
            n_states=args.n_states,
            pca_components=args.pca_components,
            random_seed=args.random_seed
        )

if __name__ == "__main__":
    main()
