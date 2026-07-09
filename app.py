import streamlit as st
import requests
import mne
import numpy as np
import glob

st.set_page_config(page_title="EEG Motor Imagery Classifier", layout="centered")

st.title("🧠 EEG Motor Imagery Classifier")
st.write("This app classifies imagined hand movement (left vs. right) from real EEG recordings, using a trained EEGNet CNN.")

mne.set_log_level('WARNING')

# Let the user pick a subject and run to load a sample epoch from
subject = st.selectbox("Choose a subject", list(range(1, 21)))
run = st.selectbox("Choose a run", [4, 8, 12])

if st.button("Load a random EEG sample and predict"):
    filepath = f'data/MNE-eegbci-data/files/eegmmidb/1.0.0/S{subject:03d}/S{subject:03d}R{run:02d}.edf'

    with st.spinner("Loading and preprocessing EEG data..."):
        raw = mne.io.read_raw_edf(filepath, preload=True)
        raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
        events, event_id = mne.events_from_annotations(raw)
        event_id_2class = {k: v for k, v in event_id.items() if k in ('T1', 'T2')}
        epochs = mne.Epochs(raw, events, event_id_2class, tmin=1.0, tmax=3.0, baseline=None, preload=True)

        X = epochs.get_data()
        y = epochs.events[:, -1]

        # Pick a random epoch from this file
        idx = np.random.randint(len(X))
        sample = X[idx]
        true_label = 'left_hand_imagery' if y[idx] == 2 else 'right_hand_imagery'

    st.success(f"Loaded epoch {idx} from subject {subject}, run {run}")
    st.write(f"**True label:** {true_label}")

    with st.spinner("Sending to model for prediction..."):
        response = requests.post(
            'http://127.0.0.1:8000/predict',
            json={'data': sample.tolist()}
        )
        result = response.json()

    st.write(f"**Predicted:** {result['prediction']}")
    st.write(f"**Confidence:** {result['confidence']}")

    st.bar_chart(result['probabilities'])

    if result['prediction'] == true_label:
        st.success("✅ Correct prediction!")
    else:
        st.error("❌ Incorrect prediction")

    # Show a snippet of the raw EEG signal for visual interest
    st.subheader("Raw EEG signal (first 5 channels)")
    st.line_chart(sample[:5].T)