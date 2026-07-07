import mne
from mne.datasets import eegbci

mne.set_log_level('DEBUG')

subjects = [1, 2, 3]
runs = [4, 8, 12]

print("Starting download...")

for subject in subjects:
    print(f"Downloading subject {subject}...")
    eegbci.load_data(subject, runs, path='./data', update_path=True)

print("Download complete!")