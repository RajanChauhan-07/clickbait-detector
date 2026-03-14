# model/classifier.py

import os
import sys
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import PROCESSED_DIR

EMBEDDINGS_DIR  = os.path.join(PROCESSED_DIR, "embeddings")
MODELS_DIR      = os.path.join(PROCESSED_DIR, "models")

FUSED_EMB_FILE  = os.path.join(EMBEDDINGS_DIR, "fused_embeddings.npy")
LABELS_FILE     = os.path.join(EMBEDDINGS_DIR, "labels.npy")
VIDEO_IDS_FILE  = os.path.join(EMBEDDINGS_DIR, "video_ids.npy")

SCALER_FILE     = os.path.join(MODELS_DIR, "scaler.joblib")
MODEL_FILE      = os.path.join(MODELS_DIR, "classifier.joblib")

LABEL_NAMES     = ["Not clickbait", "Good clickbait", "Bad clickbait"]


def train_classifier():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 65)
    print("  PHASE 14 — Classifier Training")
    print("  Model: MLP (fully connected neural network)")
    print("=" * 65 + "\n")

    # ── Load fused embeddings ─────────────────────────────────────
    print("  Loading fused embeddings...")
    X = np.load(FUSED_EMB_FILE)
    y = np.load(LABELS_FILE)
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}\n")

    # ── Train/test split ──────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y         # preserve class distribution in both splits
    )
    print(f"  Train size: {len(X_train)}")
    print(f"  Test size:  {len(X_test)}\n")

    # ── Normalize features ────────────────────────────────────────
    print("  Fitting StandardScaler...")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Compute class weights to handle imbalance ─────────────────
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    weight_dict = dict(enumerate(class_weights))
    print(f"  Class weights (balanced): {weight_dict}\n")

    # ── Build and train MLP classifier ───────────────────────────
    print("  Training MLP classifier...")
    print("  Architecture: 2049 → 512 → 256 → 128 → 3\n")

    clf = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        solver="adam",
        alpha=0.001,             # L2 regularization
        batch_size=64,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        verbose=True,
    )

    clf.fit(X_train, y_train)

    # ── Evaluate on test set ──────────────────────────────────────
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print(f"\n{'=' * 65}")
    print(f"  PHASE 15 — Evaluation Results")
    print(f"{'=' * 65}\n")
    print(f"  Accuracy:        {accuracy * 100:.2f}%")
    print(f"  F1 (macro):      {f1_macro:.4f}")
    print(f"  F1 (weighted):   {f1_weighted:.4f}")
    print(f"\n  Classification Report:")
    print(f"  {'-' * 50}")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

    print(f"  Confusion Matrix:")
    print(f"  {'-' * 50}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  {'':20}", end="")
    for name in LABEL_NAMES:
        print(f"  {name[:12]:>12}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"  {LABEL_NAMES[i][:20]:<20}", end="")
        for val in row:
            print(f"  {val:>12}", end="")
        print()

    # ── Save model and scaler ─────────────────────────────────────
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(clf,    MODEL_FILE)

    print(f"\n  [✓] Model saved  → {MODEL_FILE}")
    print(f"  [✓] Scaler saved → {SCALER_FILE}")
    print(f"\n  [✓] Phases 14 + 15 complete.\n")

    return clf, scaler, accuracy


if __name__ == "__main__":
    train_classifier()