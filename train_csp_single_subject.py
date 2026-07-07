import mne
import numpy as np
import glob
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score

mne.set_log_level('WARNING')

# Just subject 1's three runs
edf_files = sorted(glob.glob('data/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R*.edf'))
print(f"Found {len(edf_files)} EDF files for subject 1")

all_epochs = []
for filepath in edf_files:
    raw = mne.io.read_raw_edf(filepath, preload=True)
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
    events, event_id = mne.events_from_annotations(raw)
    event_id_2class = {k: v for k, v in event_id.items() if k in ('T1', 'T2')}
    epochs = mne.Epochs(raw, events, event_id_2class, tmin=1.0, tmax=3.0,
                         baseline=None, preload=True)
    all_epochs.append(epochs)

epochs_all = mne.concatenate_epochs(all_epochs)
X = epochs_all.get_data()
y = epochs_all.events[:, -1]

print(f"\nData shape: {X.shape}")
print(f"Label distribution: left={np.sum(y==2)}, right={np.sum(y==3)}")

csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)
lda = LinearDiscriminantAnalysis()
clf = Pipeline([('CSP', csp), ('LDA', lda)])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X, y, cv=cv)
print(f"\n5-fold CV accuracy (subject 1 only): {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")