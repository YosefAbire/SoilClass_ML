"""
evaluate.py — Final evaluation with all project defense artifacts.

Generates:
  - final_confusion_matrix.png  (raw counts)
  - final_confusion_matrix_norm.png  (normalised)
  - final_precision_recall_curves.png  (per-class PR curves)
  - final_f1_summary.png  (F1 bar chart + summary table)
  - Prints full classification report to stdout
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from data_loader import get_data_loaders
from losses import FocalLoss

CLASS_COLORS = ['#3A7D44', '#C47C2B', '#4A3728', '#B03A2E', '#C9A227']


def evaluate_model(model_path='soil_classifier_final.keras',
                   data_dir='soil_dataset'):

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model...")
    model = tf.keras.models.load_model(
        model_path, custom_objects={'FocalLoss': FocalLoss}
    )

    # ── Load test data ────────────────────────────────────────────────────────
    _, _, test_ds, class_names = get_data_loaders(data_dir)
    n_classes = len(class_names)

    # ── Predictions ───────────────────────────────────────────────────────────
    print("Generating predictions...")
    preds  = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = []
    for _, labels in test_ds:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_true = np.array(y_true)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    print(f"\n{'='*55}")
    print(f"Test Accuracy : {accuracy*100:.2f}%")
    print(f"Test Loss     : {loss:.4f}")
    print(f"{'='*55}")

    # ── Classification report ─────────────────────────────────────────────────
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # ── 1. Final Confusion Matrix (raw) ───────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                cmap=sns.light_palette('#3A7D44', as_cmap=True),
                xticklabels=[c.capitalize() for c in class_names],
                yticklabels=[c.capitalize() for c in class_names],
                ax=ax, linewidths=0.5, linecolor='#ddd',
                annot_kws={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_xlabel('Predicted', labelpad=10, fontsize=11)
    ax.set_ylabel('True',      labelpad=10, fontsize=11)
    ax.set_title(f'Confusion Matrix  (Accuracy: {accuracy*100:.2f}%)',
                 fontsize=13, fontweight='bold', pad=15)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('final_confusion_matrix.png', dpi=150)
    plt.close()
    print("Saved → final_confusion_matrix.png")

    # ── 2. Normalised Confusion Matrix ────────────────────────────────────────
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f',
                cmap=sns.light_palette('#B03A2E', as_cmap=True),
                xticklabels=[c.capitalize() for c in class_names],
                yticklabels=[c.capitalize() for c in class_names],
                ax=ax, linewidths=0.5, linecolor='#ddd',
                vmin=0, vmax=1,
                annot_kws={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_xlabel('Predicted', labelpad=10, fontsize=11)
    ax.set_ylabel('True',      labelpad=10, fontsize=11)
    ax.set_title('Normalised Confusion Matrix (per-class recall)',
                 fontsize=13, fontweight='bold', pad=15)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('final_confusion_matrix_norm.png', dpi=150)
    plt.close()
    print("Saved → final_confusion_matrix_norm.png")

    # ── 3. Precision-Recall Curves ────────────────────────────────────────────
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (cls, color) in enumerate(zip(class_names, CLASS_COLORS)):
        prec, rec, _ = precision_recall_curve(y_true_bin[:, i], preds[:, i])
        ap = average_precision_score(y_true_bin[:, i], preds[:, i])
        ax.plot(rec, prec, color=color, linewidth=2,
                label=f'{cls.capitalize()}  AP={ap:.2f}')
    ax.set_xlabel('Recall',    fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curves — All Classes',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('final_precision_recall_curves.png', dpi=150)
    plt.close()
    print("Saved → final_precision_recall_curves.png")

    # ── 4. F1 Summary (bar chart + table) ────────────────────────────────────
    f1_vals   = [report[c]['f1-score']  for c in class_names]
    prec_vals = [report[c]['precision'] for c in class_names]
    rec_vals  = [report[c]['recall']    for c in class_names]
    support   = [int(report[c]['support']) for c in class_names]
    macro_f1  = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']

    fig = plt.figure(figsize=(13, 6))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1], figure=fig)

    # Bar chart
    ax1 = fig.add_subplot(gs[0])
    bars = ax1.bar([c.capitalize() for c in class_names], f1_vals,
                   color=CLASS_COLORS, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, f1_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val*100:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax1.axhline(macro_f1, color='gray', linestyle='--', linewidth=1.5,
                label=f'Macro avg  {macro_f1*100:.1f}%')
    ax1.axhline(weighted_f1, color='#555', linestyle=':', linewidth=1.5,
                label=f'Weighted avg  {weighted_f1*100:.1f}%')
    ax1.set_ylim(0, 1.18)
    ax1.set_ylabel('F1 Score', fontsize=11)
    ax1.set_title('Per-class F1 Score', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Summary table
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    col_labels = ['Class', 'Precision', 'Recall', 'F1', 'Support']
    rows = [[c.capitalize(),
             f"{p*100:.1f}%", f"{r*100:.1f}%", f"{f*100:.1f}%", str(s)]
            for c, p, r, f, s in zip(class_names, prec_vals, rec_vals, f1_vals, support)]
    rows.append(['Macro avg', '',
                 '', f"{macro_f1*100:.1f}%", str(sum(support))])

    tbl = ax2.table(cellText=rows, colLabels=col_labels,
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # Style header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#3A7D44')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    # Style last row (macro avg)
    for j in range(len(col_labels)):
        tbl[len(rows), j].set_facecolor('#f0f0f0')
        tbl[len(rows), j].set_text_props(fontweight='bold')

    ax2.set_title('F1 Summary Table', fontsize=12, fontweight='bold', pad=10)

    plt.suptitle(f'Model Performance Summary  —  Accuracy: {accuracy*100:.2f}%',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('final_f1_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved → final_f1_summary.png")

    print(f"\nAll artifacts saved. Model accuracy: {accuracy*100:.2f}%")
    print(f"Macro F1: {macro_f1*100:.2f}%  |  Weighted F1: {weighted_f1*100:.2f}%")


if __name__ == "__main__":
    evaluate_model()
