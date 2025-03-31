# extract_features.py

import numpy as np
import pickle
from scipy.stats import kurtosis, skew


def time_domain_features(segmented_signals):
    """
    Compute time-domain features (mean, std, rms, etc.) for each window in each file.
    Returns a dictionary of structure:
      { filename: [ { "DE_mean": val, "DE_std": val, ... },
                    { ... },  ... ],
        ...
      }
    """
    merged_features = {}
    for filename, channels in segmented_signals.items():
        merged_features[filename] = []
        # If both DE and FE exist, combine them in one dictionary per window
        if 'DE' in channels and 'FE' in channels:
            de_windows = channels['DE']
            fe_windows = channels['FE']
            n_windows = min(len(de_windows), len(fe_windows))
            for i in range(n_windows):
                de_win = de_windows[i]
                fe_win = fe_windows[i]
                
                de_feats = {
                    "DE_mean": np.mean(de_win),
                    "DE_std": np.std(de_win),
                    "DE_rms": np.sqrt(np.mean(de_win ** 2)),
                    "DE_ptp": np.ptp(de_win),
                    "DE_crest": (np.max(np.abs(de_win)) / np.sqrt(np.mean(de_win ** 2))) if np.mean(de_win ** 2) != 0 else 0,
                    "DE_kurtosis": kurtosis(de_win),
                    "DE_skewness": skew(de_win),
                }
                
                fe_feats = {
                    "FE_mean": np.mean(fe_win),
                    "FE_std": np.std(fe_win),
                    "FE_rms": np.sqrt(np.mean(fe_win ** 2)),
                    "FE_ptp": np.ptp(fe_win),
                    "FE_crest": (np.max(np.abs(fe_win)) / np.sqrt(np.mean(fe_win ** 2))) if np.mean(fe_win ** 2) != 0 else 0,
                    "FE_kurtosis": kurtosis(fe_win),
                    "FE_skewness": skew(fe_win),
                }

                merged_dict = {**de_feats, **fe_feats}
                merged_features[filename].append(merged_dict)

        else:
            # If only DE or FE exists, compute for that channel alone
            # (In your dataset, that may be rare or never happen, but let's handle it.)
            for channel, window_array in channels.items():
                feats_list = []
                for win in window_array:
                    feats_list.append({
                        f"{channel}_mean": np.mean(win),
                        f"{channel}_std": np.std(win),
                        f"{channel}_rms": np.sqrt(np.mean(win ** 2)),
                        f"{channel}_ptp": np.ptp(win),
                        f"{channel}_crest": (np.max(np.abs(win)) / np.sqrt(np.mean(win ** 2))) if np.mean(win ** 2) != 0 else 0,
                        f"{channel}_kurtosis": kurtosis(win),
                        f"{channel}_skewness": skew(win),
                    })
                # We'll store it in the same structure
                merged_features[filename] = feats_list

    return merged_features


def frequency_domain_features(segmented_signals, fs=12000):
    """
    Compute frequency-domain features (dominant_freq, spectral_centroid, etc.)
    for each window in each file, returning a dict with the same structure:
      { filename: [ { "DE_dominant_freq": val, ... }, ... ], ... }
    """
    merged_features = {}
    for filename, channels in segmented_signals.items():
        merged_features[filename] = []
        if 'DE' in channels and 'FE' in channels:
            de_windows = channels['DE']
            fe_windows = channels['FE']
            n_windows = min(len(de_windows), len(fe_windows))
            for i in range(n_windows):
                de_win = de_windows[i]
                fe_win = fe_windows[i]

                # DE side
                fft_de = np.fft.rfft(de_win)
                mag_de = np.abs(fft_de)
                freqs_de = np.fft.rfftfreq(len(de_win), d=1/fs)

                # FE side
                fft_fe = np.fft.rfft(fe_win)
                mag_fe = np.abs(fft_fe)
                freqs_fe = np.fft.rfftfreq(len(fe_win), d=1/fs)

                de_feats = {
                    "DE_dominant_freq": freqs_de[np.argmax(mag_de)],
                    "DE_spectral_centroid": np.sum(freqs_de * mag_de) / np.sum(mag_de) if np.sum(mag_de) != 0 else 0,
                    "DE_spectral_bandwidth": np.sqrt(np.sum(mag_de * (freqs_de - (np.sum(freqs_de * mag_de) / np.sum(mag_de)))**2) / np.sum(mag_de)) if np.sum(mag_de) != 0 else 0,
                    "DE_peak_fft": np.max(mag_de),
                    "DE_total_energy": np.sum(mag_de**2),
                }

                fe_feats = {
                    "FE_dominant_freq": freqs_fe[np.argmax(mag_fe)],
                    "FE_spectral_centroid": np.sum(freqs_fe * mag_fe) / np.sum(mag_fe) if np.sum(mag_fe) != 0 else 0,
                    "FE_spectral_bandwidth": np.sqrt(np.sum(mag_fe * (freqs_fe - (np.sum(freqs_fe * mag_fe) / np.sum(mag_fe)))**2) / np.sum(mag_fe)) if np.sum(mag_fe) != 0 else 0,
                    "FE_peak_fft": np.max(mag_fe),
                    "FE_total_energy": np.sum(mag_fe**2),
                }

                merged_dict = {**de_feats, **fe_feats}
                merged_features[filename].append(merged_dict)

        else:
            # If only DE or FE, do the same approach
            for channel, window_array in channels.items():
                feats_list = []
                for win in window_array:
                    fft_vals = np.fft.rfft(win)
                    mag = np.abs(fft_vals)
                    freqs = np.fft.rfftfreq(len(win), d=1/fs)

                    feats_list.append({
                        f"{channel}_dominant_freq": freqs[np.argmax(mag)],
                        f"{channel}_spectral_centroid": (np.sum(freqs * mag) / np.sum(mag)) if np.sum(mag) != 0 else 0,
                        f"{channel}_spectral_bandwidth": np.sqrt(np.sum(mag * (freqs - (np.sum(freqs * mag) / np.sum(mag)))**2) / np.sum(mag)) if np.sum(mag) != 0 else 0,
                        f"{channel}_peak_fft": np.max(mag),
                        f"{channel}_total_energy": np.sum(mag**2),
                    })
                merged_features[filename] = feats_list

    return merged_features


def extract_and_save(
    preprocessed_path="preprocessed_data.pkl",
    feature_type="time",
    out_path="features.pkl",
    fs=12000
):
    """
    1) Load segmented signals (already preprocessed) from `preprocessed_data.pkl`.
    2) Extract features (time, freq, or both).
    3) Saves them to `features.pkl`.

    Parameters
    ----------
    preprocessed_path : str
        Path to the pickle file created by prepare_data.py
    feature_type : str
        "time" for time-domain, "freq" for frequency-domain, or "both" for merged features
    out_path : str
        Output file name (pickle) to store the features dictionary
    fs : int
        Sampling frequency for frequency-domain calculations
    """
    # 1) Load the segmented signals from pickle
    with open(preprocessed_path, 'rb') as f:
        segmented_signals = pickle.load(f)

    # 2) Extract features
    if feature_type == "time":
        feats_dict = time_domain_features(segmented_signals)
    elif feature_type == "freq":
        feats_dict = frequency_domain_features(segmented_signals, fs=fs)
    elif feature_type == "both":
        # Merge time + freq
        td = time_domain_features(segmented_signals)
        fd = frequency_domain_features(segmented_signals, fs=fs)

        feats_dict = {}
        for filename in td:
            td_list = td[filename]
            fd_list = fd.get(filename, [])
            merged_list = []
            n_windows = min(len(td_list), len(fd_list))
            for i in range(n_windows):
                combined = {}
                for k, v in td_list[i].items():
                    combined[f"TD_{k}"] = v
                for k, v in fd_list[i].items():
                    combined[f"FD_{k}"] = v
                merged_list.append(combined)
            feats_dict[filename] = merged_list
    else:
        raise ValueError("feature_type must be 'time', 'freq', or 'both'")

    # 3) Save
    with open(out_path, 'wb') as f:
        pickle.dump(feats_dict, f)
    print(f"[INFO] '{feature_type}' features saved to: {out_path}")


if __name__ == "__main__":
    # Example usage:
    extract_and_save(
        preprocessed_path="preprocessed_data.pkl",
        feature_type="time",     # or "freq" / "both"
        out_path="features_time.pkl",
        fs=12000
    )
