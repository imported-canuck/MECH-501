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
    Parameters: 
        rpm_dict (dict): Dictionary mapping descriptive keys to .npz filenames.
    Returns: 
        signals_dict (dict): Dictionary mapping descriptive keys to {'DE': array, 'FE': array}.
    Flattens each array to 1D.
    """
    signals_dict = {}
    for key, filename in rpm_dict.items():
        try:
            npz_data = np.load(filename)  # Load the .npz file
        except FileNotFoundError:
            print(f"Warning: File not found: {filename}")
            continue

        signals_dict[key] = {}
        for sensor_key in ["DE", "FE"]:
            if sensor_key in npz_data.files:
                arr = npz_data[sensor_key].flatten()  # flatten to 1D
                signals_dict[key][sensor_key] = arr
    return signals_dict

def downsample(signals_dict, factor=4):
    """
    If 'DE48' or 'Normal' is in the key, downsample from 48kHz -> 12kHz
    by taking every 'factor'-th sample. Otherwise, leave as is.
    """
    normalized_signals = {}
    for outer_key, inner_dict in signals_dict.items():
        normalized_signals[outer_key] = {}
        # Check if the outer key indicates 48 kHz or Normal (which is also 48 kHz)
        if "DE48" in outer_key or "Normal" in outer_key:
            for inner_key, inner_arr in inner_dict.items():
                normalized_signals[outer_key][inner_key] = inner_arr[::factor]
        else:
            for inner_key, inner_arr in inner_dict.items():
                normalized_signals[outer_key][inner_key] = inner_arr
    return normalized_signals

def window_signal(signal, window_size=12000, overlap=0.5):
    """
    Splits a 1D array into overlapping windows of length `window_size`.
    Overlap is a fraction (e.g. 0.5 means 50% overlap).
    """
    step = int(window_size * (1 - overlap))
    windows = []
    for start in range(0, len(signal) - window_size + 1, step):
        window = signal[start : start + window_size]
        windows.append(window)
    return np.array(windows)

def segment_signals(signals_dict, window_size=2000, overlap=0.5):
    """
    For each key in signals_dict, produce 2D arrays:
      signals_dict[key]['DE'] -> shape (#windows, window_size)
      signals_dict[key]['FE'] -> shape (#windows, window_size)
    """
    segmented_signals = {}
    for filename, channels in signals_dict.items():
        segmented_signals[filename] = {}
        for channel in ['DE', 'FE']:
            if channel in channels:
                seg = window_signal(channels[channel], window_size, overlap)
                segmented_signals[filename][channel] = seg
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
