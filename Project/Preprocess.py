'''
Preprocess.py

This script preprocesses the organized data, preparing it for feature extraction.

It uses the four dictionaires produced by load_signals in LoadSignals.py
 - signals_1730RPM
 - signals_1750RPM
 - signals_1772RPM
 - signals_1797RPM

Function "downsample" downsamples the 48 kHz signals to 12 kHz in order to maintain consistency.

Function "segment_signals" splits all of the time series signals into segments, according to a 
defined or defualt segment size and overlap.
'''

import numpy as np

def downsample(signals_dict, factor=4):
    """
    Parameters:
        signals_dict (dict): Dictionary of raw signals corresponding to filenames
        factor (int): The factor at which to downsample by (default 4)

    Returns:
        normalized_signals (dict): Dictionary of downsampled signals

    Given a dictionary containing descriptive names as keys and values:
    signals_dict[key] = {'DE': np.array, 'FE': np.array, 'BA': np.array}
    check if the data is 48 kHz (stated in the descriptive names)
    if so, downsample to 12 kHz by taking every 4th data point.

    Returns a dictionary with the same keys and downsampled np.arrays. 
    """
    normalized_signals = {}

    for outer_key, inner_dict in signals_dict.items():
        normalized_signals[outer_key] = {}
        # Check if the outer key indicates a 48 kHz signal or a Normal signal.
        if "DE48" in outer_key or "Normal" in outer_key: # Case where we downsample
            for inner_key, inner_arr in inner_dict.items():
                # Downsample the array by taking every 'factor'-th element.
                normalized_signals[outer_key][inner_key] = inner_arr[::factor]
        else: # Case where we don't
            # If condition is not met, just copy the inner arrays over.
            for inner_key, inner_arr in inner_dict.items():
                normalized_signals[outer_key][inner_key] = inner_arr
    return normalized_signals

def window_signal(signal, window_size=12000, overlap=0.5):
    """
    Parameters:
        signal (np.array): 1D array of signal data.
        window_size (int): Number of samples per window (default 12000).
        overlap (float): Fraction of overlap between consecutive windows (default 0.5).
        
    Returns:
        windows (np.array): 2D array where each row is a window of the signal.

    Splits a 1D numpy array into overlapping windows. Helper for segment_signals
    """
    # Compute the step size between windows
    step = int(window_size * (1 - overlap))
    windows = []

    for start in range(0, len(signal) - window_size + 1, step):
        window = signal[start : start + window_size] # Move window to next step
        windows.append(window) # Add the segment to the array and repeat

    return np.array(windows)

def segment_signals(signals_dict, window_size=12000, overlap=0.5):
    """
    Parameters:
        signals_dict (dict): Dictionary of signals for a specific RPM.
            Keys are descriptive names (e.g., "1730_IR_7_DE48"), and values are
            dictionaries with keys 'DE' and 'FE' and corresponding np.array signals.
        window_size (int): Number of samples per window.
        overlap (float): Fractional overlap between windows.
        
    Returns:
        segmented (dict): A nested dictionary mirroring the input structure,
            but with each channel replaced by a 2D array of windows.

    Applies windowing to all signals in a given dictionary for a particular RPM.
    
    The input dictionary (signals_dict) is assumed to have the following structure:
        { dataname1: {'DE': np.array, 'FE': np.array},
          dataname2: {'DE': np.array, 'FE': np.array},
          ... }
    where each dataname is a string like "1730_IR_7_DE48".
    
    This function processes each signal by splitting it into overlapping windows.
    """
    segmented_signals = {}

    for filename, channels in signals_dict.items(): # Iterate through files in the dict
        segmented_signals[filename] = {} # Set keys identical to original dict
        for channel in ['DE', 'FE']: 
            if channel in channels: # Just DE and FE
                # Set val of inner dict (np.array) to 2D array defined by window_signal
                segmented_signals[filename][channel] = window_signal(channels[channel], window_size, overlap)

    return segmented_signals


if __name__ == "__main__":
    example_signals = {
        "1730_B_7_DE48": {
            "DE": np.random.randn(485643),  # simulated 48 kHz drive-end signal
            "FE": np.random.randn(485763)   
        },
        "1730_Normal": {
            "DE": np.random.randn(485643),  # simulated normal file at 48 kHz
            "FE": np.random.randn(485643)
        },
        "1730_IR_7_DE12": {
            "DE": np.random.randn(121535),  # simulated file already at 12 kHz
            "FE": np.random.randn(121535)
        }
    }
    
    # Print out the shapes of the original arrays
    for desc, data_dict in example_signals.items():
        for location, arr in data_dict.items(): # Access nested dict
            print(f"{desc} location {location}: shape = {arr.shape}")

    print("\nDownsampling...")
    # Downsample the signals
    downsampled_signals = downsample(example_signals, factor=4)

    # Print out the shapes of the downsampled arrays to verify downsampling
    for desc, data_dict in downsampled_signals.items():
        for location, arr in data_dict.items():
            print(f"{desc} location {location}: shape = {arr.shape}")

    print("\nSegmenting signals with default window size and overlap...")
    # Segment all the signals in the downsampled dict
    segmented_signals = segment_signals(downsampled_signals, window_size=12000, overlap=0.5)
    
    # Print the resulting windowed segments for each file and channel
    for desc, channels in segmented_signals.items():
        for location, windows in channels.items():
            print(f"{desc} location {location}: segmented shape = {windows.shape}")
    # Confirm that the signals have been segmented into 2D np array as expected

