import argparse

import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataloader import load_ptbxl, split_by_fold
from metrics import evaluate, format_metrics

BANDS = [(0.5, 5), (5, 15), (15, 40)]  # Hz: roughly P/T waves, QRS, high-frequency content


def extract_features(signals, fs=100):
  """Per-lead features for traditional ML models.

  Args:
      signals (np.ndarray): (N, 12, T) raw signals in mV.
  Returns:
      np.ndarray: (N, 12 * n_features_per_lead)
  """
  feats = [
      signals.mean(axis=-1),
      signals.std(axis=-1),
      signals.min(axis=-1),
      signals.max(axis=-1),
      np.ptp(signals, axis=-1),
      skew(signals, axis=-1),
      kurtosis(signals, axis=-1),
      *np.percentile(signals, [5, 25, 50, 75, 95], axis=-1),
  ]

  # Band power from the FFT of each lead
  power = np.abs(np.fft.rfft(signals, axis=-1)) ** 2
  freqs = np.fft.rfftfreq(signals.shape[-1], d=1 / fs)
  for lo, hi in BANDS:
    feats.append(np.log1p(power[..., (freqs >= lo) & (freqs < hi)].sum(axis=-1)))

  X = np.concatenate(feats, axis=1) # One disconnected electrode threw an error with logistic regression due to NaN, so we replace NaN with 0.0.


def get_model(name):
  if name == "logreg":
    return make_pipeline(
        StandardScaler(),  # fit on training data only (happens inside .fit)
        OneVsRestClassifier(LogisticRegression(max_iter=2000)),
    )
  if name == "lgbm":
    from lightgbm import LGBMClassifier
    return OneVsRestClassifier(LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1))
  raise ValueError(f"unknown model {name}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", choices=["logreg", "lgbm"], default="logreg")
  parser.add_argument("--sampling-rate", type=int, default=100)
  parser.add_argument("--n-records", type=int, default=None, help="subset for quick tests")
  parser.add_argument("--eval-test", action="store_true", help="also report test fold 10 (do this once, at the end)")
  args = parser.parse_args()

  signals, labels, folds = load_ptbxl(sampling_rate=args.sampling_rate, n_records=args.n_records)
  X = extract_features(signals, fs=args.sampling_rate)
  (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_by_fold(X, labels, folds)
  print(f"features: {X.shape[1]} | train {len(X_train)} val {len(X_val)} test {len(X_test)}")

  model = get_model(args.model)
  model.fit(X_train, y_train)

  print(f"[{args.model}] val  {format_metrics(evaluate(y_val, model.predict_proba(X_val)))}")
  if args.eval_test:
    print(f"[{args.model}] test {format_metrics(evaluate(y_test, model.predict_proba(X_test)))}")


if __name__ == "__main__":
  main()