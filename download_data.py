import mne
from mne.datasets import eegbci

mne.set_log_level('WARNING')  # less noisy output now that we trust it works

# Runs 4, 8, 12 = motor imagery: left fist / right fist
subjects = list(range(1, 21))  # subjects 1 through 20
runs = [4, 8, 12]

print(f"Downloading {len(subjects)} subjects...")

for subject in subjects:
    print(f"Downloading subject {subject}...")
    eegbci.load_data(subject, runs, path='./data', update_path=True)

print("Download complete!")