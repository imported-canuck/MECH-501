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
        merged_features (dict): Dictionary with the same filenames as keys.
            For each file, a list of merged feature dictionaries is returned;
            each dictionary contains features computed from both the DE and FE signals,
            with keys prefixed by "DE_" or "FE_".
    
    This function computes the following time-domain features for each window:
      - mean, std, rms, ptp (peak-to-peak), crest (max/RMS), kurtosis, skewness.
    If both DE and FE are present, it merges their feature dictionaries per window.
    If only one channel is available, it computes features for that channel and prefixes accordingly.
    """
    merged_features = {}
    for filename, channels in segmented_signals.items():
        merged_features[filename] = []  # We want one list of merged feature dictionaries per file
        if 'DE' in channels and 'FE' in channels:
            de_windows = channels['DE']
            fe_windows = channels['FE']
            n_windows = min(len(de_windows), len(fe_windows))
            for i in range(n_windows):
                # Compute features for DE
                de_win = de_windows[i]
                de_feats = {
                    'mean': np.mean(de_win),
                    'std': np.std(de_win),
                    'rms': np.sqrt(np.mean(de_win**2)),
                    'ptp': np.ptp(de_win),
                    'crest': np.max(np.abs(de_win)) / np.sqrt(np.mean(de_win**2)) if np.sqrt(np.mean(de_win**2)) != 0 else 0,
                    'kurtosis': kurtosis(de_win),
                    'skewness': skew(de_win)
                }
                # Compute features for FE
                fe_win = fe_windows[i]
                fe_feats = {
                    'mean': np.mean(fe_win),
                    'std': np.std(fe_win),
                    'rms': np.sqrt(np.mean(fe_win**2)),
                    'ptp': np.ptp(fe_win),
                    'crest': np.max(np.abs(fe_win)) / np.sqrt(np.mean(fe_win**2)) if np.sqrt(np.mean(fe_win**2)) != 0 else 0,
                    'kurtosis': kurtosis(fe_win),
                    'skewness': skew(fe_win)
                }
                # Merge the two dictionaries with prefixes
                merged_dict = {}
                for k, v in de_feats.items():
                    merged_dict["DE_" + k] = v
                for k, v in fe_feats.items():
                    merged_dict["FE_" + k] = v
                merged_features[filename].append(merged_dict)
        else:
            # If only one channel is available, process that channel
            for channel in channels:
                feats_list = []
                for window in channels[channel]:
                    feats = {
                        'mean': np.mean(window),
                        'std': np.std(window),
                        'rms': np.sqrt(np.mean(window**2)),
                        'ptp': np.ptp(window),
                        'crest': np.max(np.abs(window)) / np.sqrt(np.mean(window**2)) if np.sqrt(np.mean(window**2)) != 0 else 0,
                        'kurtosis': kurtosis(window),
                        'skewness': skew(window)
                    }
                    # Prefix keys with the channel name
                    prefixed = { f"{channel}_{k}": v for k, v in feats.items() }
                    feats_list.append(prefixed)
                merged_features[filename] = feats_list
    return merged_features

def frequency_domain_features(segmented_signals, fs=12000):
    """
    Parameters:
        segmented_signals (dict): Dictionary structured as:
            { filename: { 'DE': 2D np.array (num_windows, window_size),
                          'FE': 2D np.array (num_windows, window_size) },
              ... }
        fs (int): Sampling frequency (default 12000 Hz).
        
    Returns:
        merged_features (dict): Dictionary where for each filename, a list of merged frequency-domain 
        feature dictionaries is returned. For each window, features computed from DE and FE are merged
        (with keys prefixed by "DE_" and "FE_").
    """
    merged_features = {}
    for filename, channels in segmented_signals.items():
        merged_features[filename] = []
        if 'DE' in channels and 'FE' in channels:
            de_windows = channels['DE']
            fe_windows = channels['FE']
            n_windows = min(len(de_windows), len(fe_windows))
            for i in range(n_windows):
                # Compute frequency features for DE
                fft_vals_de = np.fft.rfft(de_windows[i])
                mag_de = np.abs(fft_vals_de)
                freqs_de = np.fft.rfftfreq(len(de_windows[i]), d=1/fs)
                de_feats = {
                    'dominant_freq': freqs_de[np.argmax(mag_de)],
                    'spectral_centroid': np.sum(freqs_de * mag_de) / np.sum(mag_de) if np.sum(mag_de) != 0 else 0,
                    'spectral_bandwidth': np.sqrt(np.sum(mag_de * (freqs_de - (np.sum(freqs_de * mag_de) / np.sum(mag_de)))**2) / np.sum(mag_de)) if np.sum(mag_de) != 0 else 0,
                    'peak_fft': np.max(mag_de),
                    'total_energy': np.sum(mag_de**2)
                }
                # Compute frequency features for FE
                fft_vals_fe = np.fft.rfft(fe_windows[i])
                mag_fe = np.abs(fft_vals_fe)
                freqs_fe = np.fft.rfftfreq(len(fe_windows[i]), d=1/fs)
                fe_feats = {
                    'dominant_freq': freqs_fe[np.argmax(mag_fe)],
                    'spectral_centroid': np.sum(freqs_fe * mag_fe) / np.sum(mag_fe) if np.sum(mag_fe) != 0 else 0,
                    'spectral_bandwidth': np.sqrt(np.sum(mag_fe * (freqs_fe - (np.sum(freqs_fe * mag_fe) / np.sum(mag_fe)))**2) / np.sum(mag_fe)) if np.sum(mag_fe) != 0 else 0,
                    'peak_fft': np.max(mag_fe),
                    'total_energy': np.sum(mag_fe**2)
                }
                # Merge the two dictionaries with prefixes
                merged_dict = {}
                for k, v in de_feats.items():
                    merged_dict["DE_" + k] = v
                for k, v in fe_feats.items():
                    merged_dict["FE_" + k] = v
                merged_features[filename].append(merged_dict)
        else:
            # If only one channel exists, process that channel and prefix its keys.
            for channel in channels:
                feats_list = []
                for window in channels[channel]:
                    fft_vals = np.fft.rfft(window)
                    mag = np.abs(fft_vals)
                    freqs = np.fft.rfftfreq(len(window), d=1/fs)
                    feats = {
                        'dominant_freq': freqs[np.argmax(mag)],
                        'spectral_centroid': np.sum(freqs * mag) / np.sum(mag) if np.sum(mag) != 0 else 0,
                        'spectral_bandwidth': np.sqrt(np.sum(mag * (freqs - (np.sum(freqs * mag) / np.sum(mag)))**2) / np.sum(mag)) if np.sum(mag) != 0 else 0,
                        'peak_fft': np.max(mag),
                        'total_energy': np.sum(mag**2)
                    }
                    # Prefix with channel name
                    prefixed = { f"{channel}_{k}": v for k, v in feats.items() }
                    feats_list.append(prefixed)
                merged_features[filename] = feats_list
    return merged_features


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