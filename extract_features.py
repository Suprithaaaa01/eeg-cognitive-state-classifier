import mne
import numpy as np

# Load and epoch the data (same as before)
raw = mne.io.read_raw_edf('data/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R04.edf', preload=True)
events, event_id = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=4, baseline=None, preload=True)

# Define frequency bands of interest (Hz)
bands = {
    'mu': (8, 12),
    'beta': (13, 30)
}

# Get epoch data as a numpy array: shape (n_epochs, n_channels, n_timepoints)
data = epochs.get_data()
sfreq = raw.info['sfreq']

print(f"Epoch data shape: {data.shape}")  # (epochs, channels, timepoints)

# Compute band power for each epoch, channel, and band using MNE's built-in PSD function
features_list = []

for band_name, (fmin, fmax) in bands.items():
    psd, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False
    )
    # Average power across frequencies within the band -> shape (epochs, channels)
    band_power = psd.mean(axis=2)
    features_list.append(band_power)
    print(f"{band_name} band power shape: {band_power.shape}")

# Concatenate mu and beta features side by side -> shape (epochs, channels * 2)
X = np.concatenate(features_list, axis=1)
y = epochs.events[:, -1]  # the labels: 1=rest, 2=left, 3=right

print(f"\nFinal feature matrix X shape: {X.shape}")
print(f"Labels y shape: {y.shape}")
print(f"Label distribution: {np.bincount(y)}")

# Save these for the next step (training a model)
np.save('X_features.npy', X)
np.save('y_labels.npy', y)
print("\nSaved features to X_features.npy and y_labels.npy")