'''
Preprocess.py

This script preprocesses the organized data, preparing it for feature extraction.

It uses the four dictionaires produced by load_signals in LoadSignals.py
 - signals_1730RPM
 - signals_1750RPM
 - signals_1772RPM
 - signals_1797RPM

First, it downsamples the 48 kHz signals to 12 kHz in order to maintain consistency
Then, 
'''
import numpy as np

def downsample(signals_dict, factor=4):
    '''
    Inputs:
        signals_dict (dictionary)
        factor (integer)

    Output:
        normalized_signals (dictionary)

    Given a dictionary containing descriptive names as keys and values:
    signals_dict[key] = {'DE': np.array, 'FE': np.array, 'BA': np.array}
    check if the data is 48 kHz (stated in the descriptive names)
    if so, downsample to 12 kHz by taking every 4th data point.

    Returns a dictionary with the same keys and downsampled np.arrays. 
    '''
    normalized_signals = {}

    for outer_key, inner_dict in signals_dict.items():
        normalized_signals[outer_key] = {}
        # Check if the outer key indicates a 48 kHz signal or a Normal signal.
        if "DE48" in outer_key or "Normal" in outer_key:
            for inner_key, inner_arr in inner_dict.items():
                # Downsample the array by taking every 'factor'-th element.
                normalized_signals[outer_key][inner_key] = inner_arr[::factor]
        else:
            # If condition is not met, just copy the inner arrays over.
            for inner_key, inner_arr in inner_dict.items():
                normalized_signals[outer_key][inner_key] = inner_arr
    return normalized_signals







if __name__ == "__main__":
    example_signals = {
        "1730_B_7_DE48": {
            "DE": np.random.randn(485643, 1),  # simulated 48 kHz drive-end signal
            "FE": np.random.randn(485763, 1)   
        },
        "1730_Normal": {
            "DE": np.random.randn(485643, 1),  # simulated normal file at 48 kHz
            "FE": np.random.randn(485643, 1)
        },
        "1730_IR_7_DE12": {
            "DE": np.random.randn(121535, 1),  # simulated file already at 12 kHz
            "FE": np.random.randn(121535, 1)
        }
    }
    
    # Print out the shapes of the original arrays
    for desc, data_dict in example_signals.items():
        for location, arr in data_dict.items():
            print(f"{desc} location {location}: shape = {arr.shape}")

    print("Normalizing...")
    # Downsample the signals
    normalized = downsample(example_signals, factor=4)

    # Print out the shapes of the downsampled arrays to compare and verify
    for desc, data_dict in normalized.items():
        for location, arr in data_dict.items():
            print(f"{desc} location {location}: shape = {arr.shape}")



