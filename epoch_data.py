import mne

# Load the same file as before
raw = mne.io.read_raw_edf('data/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R04.edf', preload=True)

# Step 1: extract annotations (the event cues embedded in the file)
print("Annotations found in this file:")
print(raw.annotations)

# Step 2: convert annotations to MNE's numeric event format
events, event_id = mne.events_from_annotations(raw)

print("\nEvent codes (label -> number):")
print(event_id)

print("\nFirst 10 events (sample number, 0, event code):")
print(events[:10])

# Step 3: epoch the data around each event
# tmin/tmax define the time window relative to each event onset
epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=4, baseline=None, preload=True)

print("\nEpochs object summary:")
print(epochs)