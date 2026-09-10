"""Small PyTorch networks; imported only by explicit ML workflows."""
from __future__ import annotations


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is optional; install requirements-ml.txt for learned models") from exc
    return torch, nn


def build_spot_model(aux_features: int, architecture: str = "cnn_tiny"):
    torch, nn = require_torch()
    channels = (8, 16, 24) if architecture == "cnn_tiny" else (12, 24, 32)

    class ResidualSpotCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, channels[0], 3, padding=1), nn.ReLU(),
                nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1), nn.ReLU(),
                # Preserve a small spatial grid: global pooling made residual
                # correction translation-invariant and discarded subpixel cues.
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.head = nn.Sequential(nn.Linear(channels[2] * 16 + aux_features, 48), nn.ReLU(), nn.Linear(48, 4))

        def forward(self, image, auxiliary=None):
            encoded = self.encoder(image).flatten(1)
            if aux_features:
                if auxiliary is None:
                    raise ValueError("Auxiliary features are required by this checkpoint")
                encoded = torch.cat((encoded, auxiliary), dim=1)
            return self.head(encoded)

    return ResidualSpotCNN()


def build_temporal_model(kind: str, input_features: int, hidden_size: int = 24, layers: int = 1):
    torch, nn = require_torch()
    recurrent_type = nn.GRU if kind in {"gru", "gru_residual_cv"} else nn.LSTM

    class TemporalNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.recurrent = recurrent_type(input_features, hidden_size, layers, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden_size + 1, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 4))

        def forward(self, sequence, horizon):
            output, _ = self.recurrent(sequence)
            return self.head(torch.cat((output[:, -1], horizon.reshape(-1, 1)), dim=1))

    return TemporalNet()


def parameter_count(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
