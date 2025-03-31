# prepare_data.py

import numpy as np
import pickle
from BearingData import (
    files_1730RPM,
    files_1750RPM,
    files_1772RPM,
    files_1797RPM
)

def load_signals(rpm_dict):
    """
    Loads and processes signals from .npz files.

    This function takes a dictionary mapping descriptive keys to .npz filenames, 
    opens the files, and extracts the DE and FE time-series data. Each array is 
    flattened to 1D and stored in a nested dictionary structure.

    Args:
        rpm_dict (dict): A dictionary mapping descriptive keys to .npz filenames.

    Returns:
        signals_dict (dict): A nested dictionary where the outer keys are descriptive keys, and 
        the inner keys ('DE', 'FE') map to 1D numpy arrays of time-series data.
    """
    signals_dict = {}
    for key, filename in rpm_dict.items():
        try:
            npz_data = np.load(filename)  # Load the .npz file
        except FileNotFoundError:
            print(f"Warning: File not found: {filename}") # Warn if file is missing (should not happen)
            continue

        signals_dict[key] = {}
        for sensor_key in ["DE", "FE"]:
            if sensor_key in npz_data.files:
                arr = npz_data[sensor_key].flatten()  # flatten to 1D
                signals_dict[key][sensor_key] = arr
    return signals_dict

def downsample(signals_dict, factor=4):
    """
    Downsamples signals in a nested dictionary structure based on specific conditions.

    This function processes a dictionary of signals, where the outer keys represent 
    categories (e.g., "DE48", "Normal"), and the inner keys represent individual 
    signal identifiers. If the outer key contains "DE48" or "Normal", the function 
    downsamples the signals from 48 kHz to 12 kHz by taking every `factor`-th sample. 
    Otherwise, the signals are left unchanged.

    Args:
        signals_dict (dict): A nested dictionary where the outer keys are categories 
            (e.g., "DE48", "Normal"), and the inner keys map to arrays of signal data.
        factor (int, optional): The downsampling factor. Defaults to 4.

    Returns:
        normalized_signals (dict): A nested dictionary with the same structure as `signals_dict`, where 
        signals have been downsampled if their category matches the specified conditions.
    """
    normalized_signals = {}
    for outer_key, inner_dict in signals_dict.items():
        normalized_signals[outer_key] = {}
        # Check if the outer key indicates 48 kHz or Normal (which is also 48 kHz)
        if "DE48" in outer_key or "Normal" in outer_key: # Cases where downsampling should be done
            for inner_key, inner_arr in inner_dict.items(): # Downsample the signal
                normalized_signals[outer_key][inner_key] = inner_arr[::factor] # (nested dict)
        else: # Cases where downsampling should not be done
            # Copy the original signal without downsampling
            for inner_key, inner_arr in inner_dict.items():
                normalized_signals[outer_key][inner_key] = inner_arr
    return normalized_signals 

def window_signal(signal, window_size=12000, overlap=0.5): # Helper function for segment_signals
    """
    Splits a 1D signal array into overlapping windows.

    Args:
        signal (numpy.ndarray): The input 1D array to be split into windows.
        window_size (int, optional): The size of each window. Default is 12000.
        overlap (float, optional): The fraction of overlap between consecutive windows. 
            Must be between 0 and 1. Default is 0.5 (50% overlap).

    Returns:
        numpy.ndarray (np.array): A 2D array where each row corresponds to a window of the input signal.
    """
    step = int(window_size * (1 - overlap)) # Set step location
    windows = []
    for start in range(0, len(signal) - window_size + 1, step): # Segment the signal
        window = signal[start : start + window_size] # Extract the window
        windows.append(window) # Add the signal segment to the array
    return np.array(windows)

def segment_signals(signals_dict, window_size=2000, overlap=0.5):
    """
    Segments time-series signals into overlapping windows, with overlap and window size defined.

    Args:
        signals_dict (dict): A dictionary where each key is a filename, and the value is another dictionary 
            containing signal channels (e.g., 'DE', 'FE') as keys and their corresponding time-series data as values.
        window_size (int, optional): The size of each window. Defaults to 2000.
        overlap (float, optional): The fraction of overlap between consecutive windows. Defaults to 0.5.

    Returns:
        segmented_signals (dict): A dictionary with the same structure as `signals_dict`, but the values for each channel 
            are 2D arrays of segmented windows with shape (#windows, window_size).
    """   
    segmented_signals = {} # Initialize the dictionary to hold segmented signals
    # Iterate through the signals dictionary and segment each signal
    for filename, channels in signals_dict.items():
        segmented_signals[filename] = {}
        for channel in ['DE', 'FE']: # Iterate through the channels
            if channel in channels:
                seg = window_signal(channels[channel], window_size, overlap) # Call window_signal to segment the signal
                segmented_signals[filename][channel] = seg #Append the segmented signal to the dictionary
    return segmented_signals

def preprocess_and_save(
    window_size=12000,
    overlap=0.5,
    out_path='preprocessed_data.pkl'
):
    """
    1) Load signals from BearingData.
    2) Downsample them to 12 kHz.
    3) Segment them by window_size & overlap.
    4) Save the resulting dictionary (seg_signals) as a pickle.
    """
    # 1) Load all signals from the four dictionaries
    signals_1730 = load_signals(files_1730RPM)
    signals_1750 = load_signals(files_1750RPM)
    signals_1772 = load_signals(files_1772RPM)
    signals_1797 = load_signals(files_1797RPM)
    all_signals = {**signals_1730, **signals_1750, **signals_1772, **signals_1797}

    # 2) Downsample
    signals_ds = downsample(all_signals, factor=4)

    # 3) Segment
    seg_signals = segment_signals(signals_ds, window_size, overlap)

    # 4) Save
    with open(out_path, 'wb') as f:
        pickle.dump(seg_signals, f)
    print(f"[INFO] Preprocessed data saved to: {out_path}")

if __name__ == "__main__":
    # Just an example run
    preprocess_and_save(window_size=2000, overlap=0.5, out_path='preprocessed_data.pkl')
