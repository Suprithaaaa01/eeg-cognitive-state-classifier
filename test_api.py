import mne
import numpy as np
import requests
import glob

mne.set_log_level('WARNING')

# Load one file we already have locally, grab a single epoch
files = sorted(glob.glob('data/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R04.edf'))
raw = mne.io.read_raw_edf(files[0], preload=True)
raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
events, event_id = mne.events_from_annotations(raw)
event_id_2class = {k: v for k, v in event_id.items() if k in ('T1', 'T2')}
epochs = mne.Epochs(raw, events, event_id_2class, tmin=1.0, tmax=3.0, baseline=None, preload=True)

# Grab the first epoch and its true label
X = epochs.get_data()
y = epochs.events[:, -1]

sample = X[0]  # shape (64, 321)
true_label = 'left_hand_imagery' if y[0] == 2 else 'right_hand_imagery'

print(f"Sample shape: {sample.shape}")
print(f"True label: {true_label}")

# Send to the API
response = requests.post(
    'http://127.0.0.1:8000/predict',
    json={'data': sample.tolist()}
)

print("\nAPI response:")
print(response.json())