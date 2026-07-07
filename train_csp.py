import mne
import numpy as np
import glob
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score

mne.set_log_level('WARNING')

# Find all EDF files across all subjects
edf_files = sorted(glob.glob('data/MNE-eegbci-data/files/eegmmidb/1.0.0/S*/S*R04.edf')) + \
            sorted(glob.glob('data/MNE-eegbci-data/files/eegmmidb/1.0.0/S*/S*R08.edf')) + \
            sorted(glob.glob('data/MNE-eegbci-data/files/eegmmidb/1.0.0/S*/S*R12.edf'))

print(f"Found {len(edf_files)} EDF files to process")

all_epochs = []

for filepath in edf_files:
    try:
        raw = mne.io.read_raw_edf(filepath, preload=True)

        # Bandpass filter to the mu/beta range BEFORE epoching -- this is standard for CSP
        raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

        events, event_id = mne.events_from_annotations(raw)

        # Keep only left (T1) and right (T2) hand imagery -- drop rest (T0)
        event_id_2class = {k: v for k, v in event_id.items() if k in ('T1', 'T2')}

        # Focus on seconds 1-3 of each trial -- the core imagery window, skipping the initial cue-reaction period
        epochs = mne.Epochs(raw, events, event_id_2class, tmin=1.0, tmax=3.0,
                             baseline=None, preload=True)

        all_epochs.append(epochs)
        print(f"Processed {filepath}: {len(epochs)} epochs")

    except Exception as e:
        print(f"Skipping {filepath} due to error: {e}")

# Combine all subjects into one big Epochs object
epochs_all = mne.concatenate_epochs(all_epochs)

X = epochs_all.get_data()   # shape: (n_epochs, n_channels, n_times)
y = epochs_all.events[:, -1]  # labels: 2 = left (T1), 3 = right (T2)

print(f"\nCombined data shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Label distribution: left={np.sum(y==2)}, right={np.sum(y==3)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# CSP + LDA pipeline -- CSP finds spatial filters, LDA classifies in that reduced space
csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)
lda = LinearDiscriminantAnalysis()
clf = Pipeline([('CSP', csp), ('LDA', lda)])

# Cross-validate on the training set first, to get a robust sense of performance
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_train, y_train, cv=cv)
print(f"\n5-fold CV accuracy on training set: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Now fit on full training set, evaluate on held-out test set
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

print(f"\nHeld-out test accuracy: {accuracy_score(y_test, preds):.3f}")
print(classification_report(y_test, preds, target_names=['left', 'right']))

# Save the fitted pipeline for later use (e.g. in the FastAPI backend)
import joblib
joblib.dump(clf, 'csp_lda_model.pkl')
print("\nModel saved to csp_lda_model.pkl")