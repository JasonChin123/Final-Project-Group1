import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import EKGDataset, load_ptbxl, split_by_fold
from metrics import evaluate, format_metrics
from model import CNN1D, LSTMModel

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
MODELS = {"cnn": CNN1D, "lstm": LSTMModel}

# Use GPU if available
def get_device():
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def run_epoch(model, loader, loss_fn, device, optimizer=None):
  """One pass over loader. Trains if an optimizer is given, otherwise just evaluates."""
  training = optimizer is not None
  model.train(training)  # toggles dropout and batchnorm behaviour
  total_loss, all_probs, all_targets = 0.0, [], []

  with torch.set_grad_enabled(training):
    for x, y in loader:
      x, y = x.to(device), y.to(device)
      logits = model(x)
      loss = loss_fn(logits, y)

      if training:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      total_loss += loss.item() * len(x)
      all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
      all_targets.append(y.cpu().numpy())

  return total_loss / len(loader.dataset), np.concatenate(all_targets), np.concatenate(all_probs)

def augment(x, noise_std=0.02, max_mask=100, lead_drop_prob=0.1):
  """Random train-time augmentation. x: (B, 12, T), already normalized, on device."""
  B, C, T = x.shape
  x = x + noise_std * torch.randn_like(x)                         # gaussian noise

  starts = torch.randint(0, T - max_mask, (B, 1, 1), device=x.device)
  lengths = torch.randint(0, max_mask + 1, (B, 1, 1), device=x.device)
  t = torch.arange(T, device=x.device).view(1, 1, T)
  x = x.masked_fill((t >= starts) & (t < starts + lengths), 0.0)  # time mask, one gap per sample

  keep = (torch.rand(B, C, 1, device=x.device) > lead_drop_prob).to(x.dtype)
  return x * keep                                                  # lead dropout


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", choices=MODELS, default="cnn")
  parser.add_argument("--epochs", type=int, default=20)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--sampling-rate", type=int, default=100)
  parser.add_argument("--n-records", type=int, default=None, help="subset for quick tests")
  parser.add_argument("--num-workers", type=int, default=0, help="0 is fastest for in-memory data, esp. on Mac")
  parser.add_argument("--pos-weight", action="store_true", help="upweight rare classes in the loss")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--eval-test", action="store_true", help="evaluate best checkpoint on test fold 10 (once, at the end)")
  args = parser.parse_args()

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  device = get_device()

  # Data
  signals, labels, folds = load_ptbxl(sampling_rate=args.sampling_rate, n_records=args.n_records)
  (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_by_fold(signals, labels, folds)
  loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                       persistent_workers=args.num_workers > 0,  # don't respawn workers every epoch
                       pin_memory=device.type == "cuda")
  train_loader = DataLoader(EKGDataset(X_train, y_train), shuffle=True, **loader_kwargs)
  val_loader = DataLoader(EKGDataset(X_val, y_val), shuffle=False, **loader_kwargs)
  print(f"device {device} | train {len(X_train)} val {len(X_val)} test {len(X_test)}")

  # Model, loss, optimizer
  model = MODELS[args.model]().to(device)
  pos_weight = None
  if args.pos_weight:
    pos = y_train.sum(axis=0)
    pos_weight = torch.tensor((len(y_train) - pos) / np.maximum(pos, 1), dtype=torch.float32, device=device)
  loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  # Keep the checkpoint with the best validation macro AUC in training loop
  CHECKPOINT_DIR.mkdir(exist_ok=True)
  ckpt_path = CHECKPOINT_DIR / f"{args.model}_best.pt"
  best_auc = -1.0
  for epoch in range(1, args.epochs + 1):
    start = time.time()
    train_loss, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
    val_loss, val_true, val_prob = run_epoch(model, val_loader, loss_fn, device)
    metrics = evaluate(val_true, val_prob)

    marker = ""
    if metrics["macro_auc"] > best_auc:
      best_auc = metrics["macro_auc"]
      torch.save(model.state_dict(), ckpt_path)
      marker = " *"
    print(f"epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
          f"{format_metrics(metrics)} | {time.time() - start:.1f}s{marker}")

  print(f"best val macro AUC {best_auc:.3f} -> {ckpt_path}")

  if args.eval_test:
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loader = DataLoader(EKGDataset(X_test, y_test), shuffle=False, **loader_kwargs)
    test_loss, test_true, test_prob = run_epoch(model, test_loader, loss_fn, device)
    print(f"TEST | loss {test_loss:.4f} | {format_metrics(evaluate(test_true, test_prob))}")


if __name__ == "__main__":
  main()