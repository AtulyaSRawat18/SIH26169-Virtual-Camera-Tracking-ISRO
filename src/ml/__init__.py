"""Optional learned measurement-correction components (PyTorch is lazy-loaded)."""

from src.ml.correctors import CNNResidualCorrector, NoCorrection

__all__ = ["CNNResidualCorrector", "NoCorrection"]
