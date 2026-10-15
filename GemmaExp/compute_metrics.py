"""
Compute Evaluation Metrics from Gemma Prediction Logs
======================================================
Parses prediction log files, extracts ground-truth vs predicted labels
(and optional class probabilities), and computes:
- Overall Accuracy
- Macro-Averaged Precision
- Macro-Averaged Recall
- Macro-Averaged One-vs-Rest ROC-AUC
"""

import argparse
import ast
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize

LABELS_LIST = ["ν_e CC", "ν_μ CC", "Neutral Current"]


def evaluate_prediction_file(pred_file: str):
    if not os.path.exists(pred_file):
        print(f"File not found: {pred_file}")
        return

    y_true = []
    y_pred = []
    y_probs = []

    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue

            parts = [p.strip() for p in line.split("|")]
            # Format: Sample 0 | GT: ν_e CC | Pred: Neutral Current [| [0.0, 0.0, 1.0]]
            gt, pred = None, None
            for p in parts:
                if p.startswith("GT:"):
                    gt = p.replace("GT:", "").strip()
                elif p.startswith("Pred:"):
                    pred = p.replace("Pred:", "").strip()

            if gt not in LABELS_LIST or not pred:
                continue

            y_true.append(gt)
            y_pred.append(pred)

            if len(parts) >= 4:
                try:
                    probs = ast.literal_eval(parts[3])
                    y_probs.append(probs)
                except Exception:
                    pass

    if not y_true:
        print(f"No valid predictions found in: {pred_file}")
        return

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=LABELS_LIST, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=LABELS_LIST, average="macro", zero_division=0)

    print(f"Metrics for: {pred_file}")
    print(f"  Samples  : {len(y_true)}")
    print(f"  Accuracy : {acc:.4%}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")

    if len(y_probs) == len(y_true):
        try:
            y_bin = label_binarize(y_true, classes=LABELS_LIST)
            auc = roc_auc_score(y_bin, np.array(y_probs), average="macro", multi_class="ovr")
            print(f"  ROC AUC  : {auc:.4f}")
        except Exception as e:
            print(f"  ROC AUC  : N/A ({e})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compute Metrics from Prediction Files")
    parser.add_argument("--files", nargs="*", default=None,
                        help="Specific prediction file paths to analyze")
    parser.add_argument("--dir", type=str, default="./predictions",
                        help="Directory containing predictions.txt")
    args = parser.parse_args()

    if args.files:
        files = args.files
    else:
        candidates = [
            os.path.join(args.dir, "predictions.txt"),
            os.path.join(args.dir, "*", "predictions.txt"),
        ]
        import glob
        files = []
        for c in candidates:
            files.extend(glob.glob(c))

    if not files:
        print(f"No predictions files found in {args.dir}. Please specify --files.")
        return

    for file_path in files:
        evaluate_prediction_file(file_path)


if __name__ == "__main__":
    main()
