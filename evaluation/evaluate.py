# evaluation/evaluate.py

import os
import sys
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import PROCESSED_DIR

EMBEDDINGS_DIR = os.path.join(PROCESSED_DIR, "embeddings")
MODELS_DIR     = os.path.join(PROCESSED_DIR, "models")
REPORTS_DIR    = os.path.join(PROCESSED_DIR, "reports")

FUSED_EMB_FILE = os.path.join(EMBEDDINGS_DIR, "fused_embeddings.npy")
LABELS_FILE    = os.path.join(EMBEDDINGS_DIR, "labels.npy")
SCALER_FILE    = os.path.join(MODELS_DIR, "scaler.joblib")
MODEL_FILE     = os.path.join(MODELS_DIR, "classifier.joblib")

LABEL_NAMES    = ["Not Clickbait", "Good Clickbait", "Bad Clickbait"]
COLORS         = ["#2ecc71", "#f39c12", "#e74c3c"]


def evaluate():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 65)
    print("  PHASE 15 — Full Evaluation Report")
    print("=" * 65 + "\n")

    # ── Load data and model ───────────────────────────────────────
    X      = np.load(FUSED_EMB_FILE)
    y      = np.load(LABELS_FILE)
    scaler = joblib.load(SCALER_FILE)
    clf    = joblib.load(MODEL_FILE)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_test = scaler.transform(X_test)
    y_pred = clf.predict(X_test)

    # ── Metrics ───────────────────────────────────────────────────
    accuracy   = accuracy_score(y_test, y_pred)
    f1_macro   = f1_score(y_test, y_pred, average="macro")
    f1_weighted= f1_score(y_test, y_pred, average="weighted")
    precision  = precision_score(y_test, y_pred, average="macro")
    recall     = recall_score(y_test, y_pred, average="macro")
    cm         = confusion_matrix(y_test, y_pred)

    print(f"  Accuracy:       {accuracy*100:.2f}%")
    print(f"  F1 (macro):     {f1_macro:.4f}")
    print(f"  F1 (weighted):  {f1_weighted:.4f}")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:         {recall:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=LABEL_NAMES)}")

    # ── Plot ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Clickbait Detector — Evaluation Report", fontsize=18, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)

    # 1 — Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    im  = ax1.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax1)
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.set_xticklabels(LABEL_NAMES, rotation=20, ha="right", fontsize=9)
    ax1.set_yticklabels(LABEL_NAMES, fontsize=9)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Confusion Matrix")
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=13, fontweight="bold")

    # 2 — Per-class F1 bar chart
    ax2   = fig.add_subplot(gs[0, 1])
    f1_per= f1_score(y_test, y_pred, average=None)
    bars  = ax2.bar(LABEL_NAMES, f1_per, color=COLORS, edgecolor="white", linewidth=1.2)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Per-class F1 Score")
    ax2.set_xticklabels(LABEL_NAMES, rotation=15, ha="right", fontsize=9)
    for bar, val in zip(bars, f1_per):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")

    # 3 — Overall metrics bar chart
    ax3      = fig.add_subplot(gs[0, 2])
    metrics  = [accuracy, f1_macro, f1_weighted, precision, recall]
    m_labels = ["Accuracy", "F1 Macro", "F1 Weighted", "Precision", "Recall"]
    m_colors = ["#3498db", "#9b59b6", "#8e44ad", "#1abc9c", "#e67e22"]
    bars3    = ax3.bar(m_labels, metrics, color=m_colors, edgecolor="white")
    ax3.set_ylim(0, 1.1)
    ax3.set_title("Overall Metrics")
    ax3.set_xticklabels(m_labels, rotation=20, ha="right", fontsize=9)
    for bar, val in zip(bars3, metrics):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

    # 4 — Class distribution in test set
    ax4    = fig.add_subplot(gs[1, 0])
    unique, counts = np.unique(y_test, return_counts=True)
    ax4.pie(counts, labels=LABEL_NAMES, colors=COLORS,
            autopct="%1.1f%%", startangle=140,
            textprops={"fontsize": 10})
    ax4.set_title("Test Set Class Distribution")

    # 5 — Precision vs Recall per class
    ax5  = fig.add_subplot(gs[1, 1])
    prec = precision_score(y_test, y_pred, average=None)
    rec  = recall_score(y_test, y_pred, average=None)
    x    = np.arange(3)
    w    = 0.35
    ax5.bar(x - w/2, prec, w, label="Precision", color="#3498db", edgecolor="white")
    ax5.bar(x + w/2, rec,  w, label="Recall",    color="#e74c3c", edgecolor="white")
    ax5.set_xticks(x)
    ax5.set_xticklabels(LABEL_NAMES, rotation=15, ha="right", fontsize=9)
    ax5.set_ylim(0, 1.2)
    ax5.set_title("Precision vs Recall per Class")
    ax5.legend(fontsize=9)
    for i, (p, r) in enumerate(zip(prec, rec)):
        ax5.text(i - w/2, p + 0.03, f"{p:.2f}", ha="center", fontsize=9)
        ax5.text(i + w/2, r + 0.03, f"{r:.2f}", ha="center", fontsize=9)

    # 6 — Summary text box
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    summary = (
        f"CLICKBAIT DETECTOR\n"
        f"{'─' * 30}\n\n"
        f"Dataset:       1177 videos\n"
        f"Test set:       {len(y_test)} videos\n\n"
        f"Accuracy:      {accuracy*100:.2f}%\n"
        f"F1 Macro:      {f1_macro:.4f}\n"
        f"F1 Weighted:   {f1_weighted:.4f}\n"
        f"Precision:     {precision:.4f}\n"
        f"Recall:        {recall:.4f}\n\n"
        f"Architecture:\n"
        f"  Text: MiniLM-L6-v2\n"
        f"  Image: CLIP ViT-B/32\n"
        f"  Fusion: Concatenation\n"
        f"  Classifier: MLP\n"
        f"  Input dim: 2049\n\n"
        f"Classes:\n"
        f"  0 - Not Clickbait\n"
        f"  1 - Good Clickbait\n"
        f"  2 - Bad Clickbait"
    )
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
             fontsize=10, verticalalignment="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#f8f9fa", alpha=0.8))

    # ── Save plot ─────────────────────────────────────────────────
    plot_path = os.path.join(REPORTS_DIR, "evaluation_report.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\n  [✓] Report saved → {plot_path}")
    print(f"  [✓] Phase 15 complete.")
    print(f"\n  {'=' * 65}")
    print(f"  ALL 15 PHASES COMPLETE — Clickbait Detector built end to end.")
    print(f"  {'=' * 65}\n")


if __name__ == "__main__":
    evaluate()