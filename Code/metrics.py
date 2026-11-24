import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from dataloader import CLASSES


def evaluate(y_true, y_prob, class_names=CLASSES):
  """Per-class and macro ROC-AUC / PR-AUC for multi-label predictions.

  Args:
      y_true (np.ndarray): (N, C) multi-label ground truth.
      y_prob (np.ndarray): (N, C) predicted probabilities (sigmoid).
  """
  results = {}
  aucs, aps = [], []
  for i, name in enumerate(class_names):
    if len(np.unique(y_true[:, i])) < 2:  # AUC is undefined if a class is all 0s or all 1s
      auc = ap = float("nan")
    else:
      auc = roc_auc_score(y_true[:, i], y_prob[:, i])
      ap = average_precision_score(y_true[:, i], y_prob[:, i])
    results[f"auc_{name}"] = auc
    results[f"ap_{name}"] = ap
    aucs.append(auc)
    aps.append(ap)

  results["macro_auc"] = float(np.nanmean(aucs))
  results["macro_ap"] = float(np.nanmean(aps))
  return results


def format_metrics(metrics, class_names=CLASSES):
  per_class = " ".join(f"{n}={metrics[f'auc_{n}']:.3f}" for n in class_names)
  return (f"macro AUC {metrics['macro_auc']:.3f} | macro PR-AUC {metrics['macro_ap']:.3f}" f" | AUC per class: {per_class}")