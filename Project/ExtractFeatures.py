'''
ExtractFeatures.py

This script extracts time-domain and frequency-domain features from segmented data

It uses the dictionary produced by "downsample" and "segment_signals" in Preprocess.py

The features are stored in a doubly nested dictionary within a list. Each
window (chunk of split data) turns into one dictionary with tags of featutres 
as keys and feature values as corresponding values. Functions "time_domain_features"
and "frequency_domain_features" does this for time, and frequency domain 
features respectively. 
'''

import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler

def time_domain_features(segmented_signals):
    """
    Parameters:
        segmented_signals (dict): Dictionary structured as:
            { filename: { 'DE': 2D np.array (num_windows, window_size),
                          'FE': 2D np.array (num_windows, window_size) },
              ... }
              
    Returns:
        features_dict (dict): Dictionary with the same keys as segmented_signals.
            Each channel (DE, FE) will have a list of dictionaries, one per window,
            containing the computed features.
    
        Extract time-domain features for each window in the segmented signals.
    
    For each window, the following time-domain features are computed:
      - Mean
      - Standard Deviation
      - RMS (Root Mean Square)
      - Peak-to-Peak (max-min)
      - Crest Factor (max / RMS)
      - Kurtosis
      - Skewness
    """
    features_dict = {}
    for filename, channels in segmented_signals.items():
        features_dict[filename] = {} # Set outer keys as same as in segmented_signals
        for channel in ['DE', 'FE']:
            if channel in channels: # Safeguard to only process channels that exist
                windows = channels[channel]
                channel_features = []
                # Process each window (row in the 2D array)
                for window in windows:
                    # Compute time-domain features
                    mean_val = np.mean(window)
                    std_val = np.std(window)
                    rms_val = np.sqrt(np.mean(window**2))
                    ptp_val = np.ptp(window)  # Peak-to-peak value
                    crest = (np.max(np.abs(window)) / rms_val) if rms_val != 0 else 0 # C = Peak/RMS
                    kurt = kurtosis(window)
                    skewness = skew(window)
                    
                    # Store the computed features in a dictionary
                    feat_dict = {
                        'mean': mean_val,
                        'std': std_val,
                        'rms': rms_val,
                        'ptp': ptp_val,
                        'crest': crest,
                        'kurtosis': kurt,
                        'skewness': skewness
                    }
                    channel_features.append(feat_dict)
                features_dict[filename][channel] = channel_features
    return features_dict

def frequency_domain_features(segmented_signals, fs=12000):
    """  
    Parameters:
        segmented_signals (dict): Dictionary structured as:
            { filename: { 'DE': 2D np.array (num_windows, window_size),
                          'FE': 2D np.array (num_windows, window_size) },
              ... }
        fs (int): Sampling frequency (default 12000 Hz).
        
    Returns:
        features_dict (dict): Dictionary with the same keys as segmented_signals.
            Each channel (DE, FE) will have a list of dictionaries, one per window,
            containing the computed frequency-domain features.

    Extract frequency-domain features for each window in the segmented signals.
    
    For each window, the following features are computed:
      - Dominant Frequency: Frequency at maximum FFT magnitude.
      - Spectral Centroid: Weighted average frequency.
      - Spectral Bandwidth: Spread of the spectrum around the centroid.
      - Peak FFT Amplitude: Maximum amplitude in the FFT spectrum.
      - Total Spectral Energy: Sum of squared FFT magnitudes.
    """
    features_dict = {}
    for filename, channels in segmented_signals.items():
        features_dict[filename] = {}
        for channel in ['DE', 'FE']:
            if channel in channels: # Safeguard to only process channels that exist
                windows = channels[channel]
                channel_features = []
                for window in windows:
                    # Compute FFT on the window (using real FFT)
                    fft_vals = np.fft.rfft(window)
                    mag = np.abs(fft_vals)
                    
                    # Frequency bins corresponding to the FFT values
                    freqs = np.fft.rfftfreq(len(window), d=1/fs)
                    
                    # Dominant Frequency: frequency at maximum magnitude
                    dominant_freq = freqs[np.argmax(mag)]
                    
                    # Spectral Centroid: weighted average of frequencies
                    spectral_centroid = np.sum(freqs * mag) / np.sum(mag) if np.sum(mag) != 0 else 0
                    
                    # Spectral Bandwidth: weighted standard deviation around the centroid
                    spectral_bandwidth = np.sqrt(np.sum(mag * (freqs - spectral_centroid)**2) / np.sum(mag)) if np.sum(mag) != 0 else 0
                    
                    # Peak FFT Amplitude: maximum amplitude in the FFT
                    peak_fft = np.max(mag)
                    
                    # Total Spectral Energy: sum of squared FFT magnitudes
                    total_energy = np.sum(mag**2)
                    
                    feat_dict = {
                        'dominant_freq': dominant_freq,
                        'spectral_centroid': spectral_centroid,
                        'spectral_bandwidth': spectral_bandwidth,
                        'peak_fft': peak_fft,
                        'total_energy': total_energy
                    }
                    channel_features.append(feat_dict)
                features_dict[filename][channel] = channel_features
    return features_dict

if __name__ == "__main__":
    # Test the feature extraction functions with simulated segmented signals.
    simulated_segmented_signals = {
        "Test_File": {
            "DE": [np.random.randn(12000), np.random.randn(12000)],
            "FE": [np.random.randn(12000), np.random.randn(12000)]
        }
    }
    
    td_feats = time_domain_features(simulated_segmented_signals)
    fd_feats = frequency_domain_features(simulated_segmented_signals, fs=12000)
    
    print("Time-domain features for Test_File, DE:")
    print(td_feats["Test_File"]["DE"])
    print("\nFrequency-domain features for Test_File, DE:")
    print(fd_feats["Test_File"]["DE"])