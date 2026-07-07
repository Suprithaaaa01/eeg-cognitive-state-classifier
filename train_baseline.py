import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load our saved features
X = np.load('X_features.npy')
y = np.load('y_labels.npy')

print(f"Loaded X: {X.shape}, y: {y.shape}")

# Split into train/test sets (80/20), stratified to keep class balance in both
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Standardize features (important for SVMs especially)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model 1: SVM ---
print("\n=== SVM ===")
svm = SVC(kernel='rbf', C=1.0)
svm.fit(X_train_scaled, y_train)
svm_preds = svm.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, svm_preds):.3f}")
print(classification_report(y_test, svm_preds, target_names=['rest', 'left', 'right']))

# --- Model 2: Random Forest ---
print("\n=== Random Forest ===")
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_preds = rf.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.3f}")
print(classification_report(y_test, rf_preds, target_names=['rest', 'left', 'right']))