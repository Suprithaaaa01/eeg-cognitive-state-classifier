import matplotlib
matplotlib.use('MacOSX')  # force the native Mac GUI backend
import matplotlib.pyplot as plt
import mne

raw = mne.io.read_raw_edf('data/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R04.edf', preload=True)

print(raw.info)

raw.plot(duration=10, n_channels=20, scalings='auto', block=True)