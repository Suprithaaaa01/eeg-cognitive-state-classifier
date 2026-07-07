import mne
import numpy as np
import glob
import re
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score

mne.set_log_level('WARNING')

def load_subject_epochs(subject_num):
    """Load and epoch all 3 runs for one subject, return combined Epochs object."""
    pattern = f'data/MNE-eegbci-data/files/eegmmidb/1.0.0/S{subject_num:03d}/S{subject_num:03d}R*.edf'
    files = sorted(glob.glob(pattern))
    subject_epochs = []
    for filepath in files:
        raw = mne.io.read_raw_edf(filepath, preload=True)
        raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
        events, event_id = mne.events_from_annotations(raw)
        event_id_2class = {k: v for k, v in event_id.items() if k in ('T1', 'T2')}
        epochs = mne.Epochs(raw, events, event_id_2class, tmin=1.0, tmax=3.0,
                             baseline=None, preload=True)
        subject_epochs.append(epochs)
    return mne.concatenate_epochs(subject_epochs)

# Subject-level split: train on 1-16, test entirely on unseen subjects 17-20
train_subjects = list(range(1, 17))
test_subjects = list(range(17, 21))

print(f"Train subjects: {train_subjects}")
print(f"Test subjects (never seen during training): {test_subjects}")

print("\nLoading training subjects...")
train_epochs_list = [load_subject_epochs(s) for s in train_subjects]
train_epochs = mne.concatenate_epochs(train_epochs_list)

print("Loading test subjects...")
test_epochs_list = [load_subject_epochs(s) for s in test_subjects]
test_epochs = mne.concatenate_epochs(test_epochs_list)

X_train = train_epochs.get_data()
y_train = train_epochs.events[:, -1]
X_test = test_epochs.get_data()
y_test = test_epochs.events[:, -1]

print(f"\nTrain shape: {X_train.shape}, label dist: left={np.sum(y_train==2)}, right={np.sum(y_train==3)}")
print(f"Test shape: {X_test.shape}, label dist: left={np.sum(y_test==2)}, right={np.sum(y_test==3)}")

# CSP + LDA
csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)
lda = LinearDiscriminantAnalysis()
clf = Pipeline([('CSP', csp), ('LDA', lda)])

clf.fit(X_train, y_train)
preds = clf.predict(X_test)

print(f"\nSubject-wise held-out test accuracy: {accuracy_score(y_test, preds):.3f}")
print(classification_report(y_test, preds, target_names=['left', 'right']))

import joblib
joblib.dump(clf, 'csp_lda_subjectwise_model.pkl')
print("\nModel saved to csp_lda_subjectwise_model.pkl")