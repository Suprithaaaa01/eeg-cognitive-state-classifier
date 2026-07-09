#  EEG Motor Imagery Classifier

A full-stack machine learning system that classifies imagined hand movement (left vs. right) from real human EEG recordings — from raw brain signal to a live, interactive prediction interface.

Built on the [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) (109-subject, 64-channel EEG recordings), this project benchmarks classical BCI signal processing against a custom deep learning model, and investigates a real, unsolved challenge in brain-computer interfaces: **how well do models trained on one group of people generalize to a person they've never seen?**

##  Key Finding

| Approach | Setup | Accuracy | What it tells us |
|---|---|---|---|
| CSP + LDA | Single-subject (train/test same person) | **66.7%** | Matches published BCI literature — the pipeline works correctly |
| CSP + LDA | Cross-subject (naive pooled split) | 45.6% | Misleadingly low — epoch-level splitting leaks subject identity across train/test |
| CSP + LDA | Cross-subject (proper subject-wise split) | **51.7%** | The honest generalization number — barely above chance |
| EEGNet (CNN) | Cross-subject, early-stopped | **55.6%** | Deep learning offers a modest edge, but the same fundamental limit persists |

**The takeaway:** both a classical signal-processing pipeline (CSP+LDA) and a purpose-built CNN (EEGNet) struggle to generalize motor imagery classification to unseen subjects, despite performing well within a single subject. This mirrors a well-documented open problem in BCI research — inter-subject variability in EEG signal — rather than a flaw in either model. This project treats that "negative" result as a legitimate finding, not a failure, and the codebase is structured specifically to let you *see* the difference between a naive and a rigorous evaluation setup.

##  Architecture

```
Raw EEG (.edf files, PhysioNet)
        │
        ▼
MNE-Python preprocessing (bandpass filter 7–30Hz, epoching)
        │
        ├──► Band power features ──► SVM / Random Forest baseline
        │
        ├──► CSP spatial filters ──► LDA classifier
        │
        └──► Raw epoched signal ──► EEGNet CNN (PyTorch, trained on GPU)
                                            │
                                            ▼
                                  FastAPI backend (/predict)
                                            │
                                            ▼
                                  Streamlit frontend (live demo)
```

## Tech Stack

- **Signal processing:** MNE-Python (industry-standard neuroscience toolkit)
- **Classical ML:** scikit-learn (SVM, Random Forest, CSP, LDA)
- **Deep learning:** PyTorch — custom EEGNet implementation (Lawhern et al., 2018), trained on Google Colab GPU
- **Backend:** FastAPI — serves the trained model via a REST `/predict` endpoint
- **Frontend:** Streamlit — interactive demo for loading real EEG samples and viewing live predictions
- **Data:** [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) — 64-channel EEG, 20 subjects, motor imagery tasks

## What's in this repo

| File | Purpose |
|---|---|
| `download_data.py` | Downloads EEG recordings for 20 subjects from PhysioNet via MNE |
| `explore_data.py` | Loads and visualizes a raw EEG recording |
| `epoch_data.py` | Extracts event markers and epochs the continuous signal |
| `extract_features.py` | Computes mu/beta band power features across all subjects |
| `train_baseline.py` | Classical ML baseline (SVM, Random Forest) on band power features |
| `train_csp.py` | CSP + LDA pipeline, naive cross-subject split |
| `train_csp_single_subject.py` | CSP + LDA, single-subject validation (sanity check) |
| `train_csp_subjectwise.py` | CSP + LDA, proper subject-wise train/test split |
| `eegnet_training.ipynb` | EEGNet CNN architecture and training (Google Colab, GPU) |
| `eegnet_best_model.pt` | Trained EEGNet weights (early-stopped at peak validation accuracy) |
| `main.py` | FastAPI backend serving live predictions |
| `app.py` | Streamlit frontend for interactive demo |
| `test_api.py` | Script to test the API with a real EEG sample |

## Running it locally

**1. Clone and set up the environment**
```bash
git clone https://github.com/Suprithaaaa01/eeg-cognitive-state-classifier.git
cd eeg-cognitive-state-classifier
python3.11 -m venv venv
source venv/bin/activate
pip install mne mne-bids scikit-learn torch fastapi uvicorn streamlit requests joblib
```

**2. Download the dataset**
```bash
python3.11 download_data.py
```

**3. Start the backend** (in one terminal)
```bash
uvicorn main:app --reload
```

**4. Start the frontend** (in a second terminal)
```bash
streamlit run app.py
```

Then open `http://localhost:8501` and try loading different subjects/runs to see live predictions against ground truth.

## Methodology notes

- **Frequency band:** Signal was bandpass filtered to 7–30 Hz (mu and beta bands), the range most associated with motor imagery-related desynchronization (ERD).
- **Epoching window:** Classification uses seconds 1–3 of each 4-second trial, skipping the initial cue-reaction period.
- **CSP:** 6 spatial components, following standard practice in BCI competition literature.
- **EEGNet:** A compact CNN (~2,450 parameters) purpose-built for EEG, using temporal → depthwise spatial → separable convolutions. Trained with early stopping on held-out subject validation accuracy to control for overfitting given the small dataset size.
- **Evaluation:** All headline results use a **subject-wise split** (train on subjects 1–16, test on entirely unseen subjects 17–20) — the scientifically appropriate way to measure real-world generalization, as opposed to randomly splitting individual epochs (which can leak subject-specific signal between train and test sets).

##  Background

Motor imagery BCI classification is a well-studied problem in computational neuroscience — imagining a movement (without executing it) produces measurable, lateralized changes in sensorimotor cortex oscillatory power, a phenomenon called event-related desynchronization (ERD). This project's results are broadly consistent with published benchmarks on this exact dataset, and the cross-subject generalization gap observed here reflects a genuine, active area of BCI research (often addressed via transfer learning, subject-specific calibration, or larger multi-subject training sets than used here).

##  Future improvements

- Riemannian geometry-based classifiers (often outperform CSP for cross-subject BCI)
- Subject-specific fine-tuning / calibration layer
- Larger training set (all 109 subjects)
- Data augmentation for EEG (time warping, channel dropout)
- Docker containerization and cloud deployment