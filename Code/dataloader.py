import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import DataLoader, Dataset

DATA_DIR = (Path(__file__).resolve().parent.parent / "Data"
            / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1")
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]  # fixed order of our 5 classes of interest (normal, myocardial infarction, ST/T change, conduction disturbance, hypertrophy)


def load_ptbxl(data_dir=DATA_DIR, sampling_rate=100, n_records=None):
  """Returns signals (N, 12, T) float32, multi-hot labels (N, 5) float32, folds (N,)."""
  meta = pd.read_csv(data_dir / "ptbxl_database.csv", index_col="ecg_id")
  meta["scp_codes"] = meta["scp_codes"].apply(ast.literal_eval)  # string -> dict

  scp = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)
  scp = scp[scp["diagnostic"] == 1]

  def to_superclasses(codes):
    return {scp.loc[c, "diagnostic_class"] for c in codes if c in scp.index}

  meta["superclasses"] = meta["scp_codes"].apply(to_superclasses)
  meta = meta[meta["superclasses"].apply(len) > 0]  # drop records with no diagnostic superclass

  if n_records is not None:
    meta = meta.iloc[:n_records]

  labels = np.array([[c in s for c in CLASSES] for s in meta["superclasses"]], dtype=np.float32)

  col = "filename_lr" if sampling_rate == 100 else "filename_hr"
  signals = np.stack([wfdb.rdsamp(str(data_dir / f))[0].astype(np.float32) for f in meta[col]])
  signals = signals.transpose(0, 2, 1)  # (N, T, 12) -> (N, 12, T) for Conv1d

  return signals, labels, meta["strat_fold"].to_numpy()


def split_by_fold(signals, labels, folds):
  """PTB-XL recommended split: folds 1-8 train, 9 val, 10 test (patient-separated so as not to have data leakage; https://physionet.org/content/ptb-xl/1.0.1/)."""
  train, val, test = folds <= 8, folds == 9, folds == 10
  return (signals[train], labels[train]), (signals[val], labels[val]), (signals[test], labels[test])


class EKGDataset(Dataset):

  def __init__(self, signals, labels, transform=None):
    """
        Args:
            signals (np.ndarray): Shape (N, Sequence_Length) for 1-lead 
                                  or (N, Channels, Sequence_Length) for multi-lead.
            labels (np.ndarray): Shape (N, 5) for classification.
            transform (callable, optional): Optional augmentations/filters.
        """
    self.signals = signals
    self.labels = labels
    self.transform = transform

  def __len__(self):
    return len(self.signals)

  def __getitem__(self, idx):
    signal = self.signals[idx]
    label = self.labels[idx]

    # Ensure shape is (Channels, Sequence_Length)
    if signal.ndim == 1:
      signal = np.expand_dims(signal, axis=0)  # Becomes (1, Sequence_Length)

    # Convert to float32 tensor (standard for PyTorch model weights)
    signal_tensor = torch.tensor(signal, dtype=torch.float32)

    # Per-lead Z-score normalization (helps when using different models and across different data sources)
    mean = signal_tensor.mean(dim=-1, keepdim=True)
    std = signal_tensor.std(dim=-1, keepdim=True) + 1e-8
    signal_tensor = (signal_tensor - mean) / std

    if self.transform:
      signal_tensor = self.transform(signal_tensor)

    label_tensor = torch.tensor(label, dtype=torch.float32)

    return signal_tensor, label_tensor

if __name__ == "__main__":
  
  signals, labels, folds = load_ptbxl(sampling_rate=100, n_records=500)  # small subset for testing

  (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_by_fold(signals, labels, folds)
  print("train", X_train.shape, y_train.shape, "val", X_val.shape, "test", X_test.shape)
  print("label counts:", dict(zip(CLASSES, y_train.sum(axis=0).tolist())))

  train_loader = DataLoader(
      EKGDataset(X_train, y_train),
      batch_size=32,
      shuffle=True,
      num_workers=2,                          
      pin_memory=torch.cuda.is_available(),   # only useful with a GPU; warns on Mac
  )

  batch_x, batch_y = next(iter(train_loader))
  print("batch", batch_x.shape, batch_y.shape)  # expect [32, 12, 1000] and [32, 5]