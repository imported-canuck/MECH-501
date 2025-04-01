'''
LoadSignals.py

This script loads bearing vibration data from four dictionaries of filepaths:
 - files_1730RPM
 - files_1750RPM
 - files_1772RPM
 - files_1797RPM

Each dictionary key is a descriptive string, e.g. '1730_B_7_DE48', and its value
is the name of the .npz file. The script loads each .npz, checks for 'DE', 'FE'
arrays, and stores the flattened data in a signals dictionary. Disregards 'BA' signals

The function is used to produce four dictionaries:
 - signals_1730RPM
 - signals_1750RPM
 - signals_1772RPM
 - signals_1797RPM
where each is a nested dict: { same key : {'DE': array, 'FE': array} }
Each dict has keys corresponding to descriptive names (same keys as bearing data dict)
Corresponding values is a dict containing the tags of the location ('DE', 'FE') and the time-series vibration data at those locations.
'''

import numpy as np

# Import the file dictionaries from 'Bearing Data.py'
# Make sure 'Bearing Data.py' is in the same folder or adjust the import path
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
        signals_dict (dict): Dictionary mapping descriptive keys to dictionary
            containing channel tags and time-series data.

    Given a dictionary mapping descriptive keys to .npz filenames, 
    load the data and return a dict of:
        signals_dict[key] = {'DE': np.array, 'FE': np.array}
    Keys are same as those in rpm_dict

    Flattening is done so each array becomes 1D.
    """
    signals_dict = {}
    for key, filename in rpm_dict.items():
        try:
            npz_data = np.load(filename)  # Load the .npz file
        except FileNotFoundError: # Catch but don't stop if a file not found. This shouldn't happen.
            print(f"Warning: File not found: {filename}")
            continue

        # Create a sub-dict
        signals_dict[key] = {} # Set key same as rpm_dict
        # Each .npz might (should) have 'DE', 'FE'
        for sensor_key in ["DE", "FE"]:
            if sensor_key in npz_data.files: # The key is the sensor name ("DE", "FE")
                arr = npz_data[sensor_key] # The value is the time-series data array

                # Flatten to 1D array since shape is (N,1)
                arr = arr.flatten()
                signals_dict[key][sensor_key] = arr # Since its a nested dictionary
    
    return signals_dict


if __name__ == "__main__": 
    print("Loading 1730 RPM files...")
    signals_1730RPM = load_signals(files_1730RPM)

    print("\nLoading 1750 RPM files...")
    signals_1750RPM = load_signals(files_1750RPM)

    print("\nLoading 1772 RPM files...")
    signals_1772RPM = load_signals(files_1772RPM)

    print("\nLoading 1797 RPM files...")
    signals_1797RPM = load_signals(files_1797RPM)

    # Example, check shapes (# of data points)
    for key, sensor_dict in signals_1730RPM.items():
        if 'DE' in sensor_dict:
            print(f"File: {key}, DE shape: {sensor_dict['DE'].shape}")
        if 'FE' in sensor_dict:
            print(f"File: {key}, FE shape: {sensor_dict['FE'].shape}")
        if 'BA' in sensor_dict:
            print(f"File: {key}, BA shape: {sensor_dict['BA'].shape}")

    print("\nDone loading data")