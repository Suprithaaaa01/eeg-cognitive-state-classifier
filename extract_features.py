import mne
import numpy as np
import glob

mne.set_log_level('WARNING')

bands = {
    'mu': (8, 12),
    'beta': (13, 30)
}

all_X = []
all_y = []

# Find all downloaded EDF files for runs 4, 8, 12 across all subjects
edf_files = sorted(glob.glob('data/MNE-eegbci-data/files/eegmmidb/1.0.0/S*/S*R04.edf')) + \
            sorted(glob.glob('data/MNE-eegbci-data/files/eegmmidb/1.0.0/S*/S*R08.edf')) + \
            sorted(glob.glob('data/MNE-eegbci-data/files/eegmmidb/1.0.0/S*/S*R12.edf'))

print(f"Found {len(edf_files)} EDF files to process")

for filepath in edf_files:
    try:
        raw = mne.io.read_raw_edf(filepath, preload=True)
        events, event_id = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=4, baseline=None, preload=True)

        data = epochs.get_data()
        sfreq = raw.info['sfreq']

        features_list = []
        for band_name, (fmin, fmax) in bands.items():
            psd, freqs = mne.time_frequency.psd_array_welch(
                data, sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False
            )
            band_power = psd.mean(axis=2)
            features_list.append(band_power)

        X = np.concatenate(features_list, axis=1)
        y = epochs.events[:, -1]

        all_X.append(X)
        all_y.append(y)
        print(f"Processed {filepath}: {X.shape[0]} epochs")

    except Exception as e:
        print(f"Skipping {filepath} due to error: {e}")

# Combine everything into one big dataset
X_all = np.concatenate(all_X, axis=0)
y_all = np.concatenate(all_y, axis=0)

print(f"\nFinal combined feature matrix: {X_all.shape}")
print(f"Final labels: {y_all.shape}")
print(f"Label distribution: {np.bincount(y_all)}")

np.save('X_features.npy', X_all)
np.save('y_labels.npy', y_all)
print("\nSaved combined features to X_features.npy and y_labels.npy")