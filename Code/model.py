import torch
import torch.nn as nn


class CNN1D(nn.Module):
  """Stack of Conv1d blocks -> global average pool -> linear head.

  Input:  (batch, 12, T)   leads as channels
  Output: (batch, num_classes) raw logits (use BCEWithLogitsLoss, sigmoid only for metrics)
  """

  def __init__(self, in_channels=12, num_classes=5, channels=(32, 64, 128, 128),
               kernel_size=7, dropout=0.3):
    super().__init__()
    layers = []
    prev = in_channels
    for ch in channels:
      layers += [
          nn.Conv1d(prev, ch, kernel_size, padding=kernel_size // 2, bias=False),  # padding keeps length
          nn.BatchNorm1d(ch),
          nn.ReLU(),
          nn.MaxPool1d(2),  # halves the time dimension
      ]
      prev = ch
    self.features = nn.Sequential(*layers)
    self.pool = nn.AdaptiveAvgPool1d(1)  # (batch, C, T') -> (batch, C, 1), works for any T
    self.head = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(prev, num_classes),
    )

  def forward(self, x):
    return self.head(self.pool(self.features(x)))


class LSTMModel(nn.Module):
  """Optional conv front-end (downsamples time by 4) -> (bi)LSTM -> mean over time -> linear head.

  Input:  (batch, 12, T)   same layout as the CNN, permuted internally
  Output: (batch, num_classes) raw logits
  """

  def __init__(self, in_channels=12, num_classes=5, hidden_size=128, num_layers=2,
               bidirectional=True, downsample=True, dropout=0.3):
    super().__init__()
    if downsample:
      self.frontend = nn.Sequential(
          nn.Conv1d(in_channels, 64, 7, padding=3, bias=False), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
          nn.Conv1d(64, 64, 7, padding=3, bias=False), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
      )
      lstm_in = 64
    else:
      self.frontend = nn.Identity()
      lstm_in = in_channels

    self.lstm = nn.LSTM(
        input_size=lstm_in,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        bidirectional=bidirectional,
        dropout=dropout if num_layers > 1 else 0.0,
    )
    out_dim = hidden_size * (2 if bidirectional else 1)
    self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, num_classes))

  def forward(self, x):
    x = self.frontend(x)      # (batch, C, T')
    x = x.permute(0, 2, 1)    # (batch, T', C): batch_first LSTM wants time before features
    out, _ = self.lstm(x)     # (batch, T', hidden * directions)
    return self.head(out.mean(dim=1))  # mean-pool over time instead of only the last step


if __name__ == "__main__":
  x = torch.randn(8, 12, 1000) # Random batch to init and check
  for model in (CNN1D(), LSTMModel()):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{type(model).__name__}: out {tuple(model(x).shape)}, {n_params:,} params")