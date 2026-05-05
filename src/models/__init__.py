"""
Modeling Module
===============

Provides machine learning models for anomaly detection and conflict prediction.

Classes:
    AnomalyModel: Unsupervised anomaly detection (Isolation Forest, VAE, DBSCAN)
    ConflictPredictor: Conflict prediction (LSTM-Attention, TFT, XGBoost)
    BaselineModel: Statistical baselines (ARIMA, Prophet)
    ModelEvaluator: Model evaluation and reporting
"""


def __getattr__(name):
    """Lazy import to avoid RuntimeWarning with python -m."""
    if name == "AnomalyModel":
        from .anomaly_model import AnomalyModel
        return AnomalyModel
    elif name == "IsolationForestDetector":
        from .anomaly_model import IsolationForestDetector
        return IsolationForestDetector
    elif name == "VAEDetector":
        from .anomaly_model import VAEDetector
        return VAEDetector
    elif name == "ConflictPredictor":
        from .conflict_predictor import ConflictPredictor
        return ConflictPredictor
    elif name == "ConflictLSTM":
        from .conflict_predictor import ConflictLSTM
        return ConflictLSTM
    elif name == "BaselineModel":
        from .baseline import BaselineModel
        return BaselineModel
    elif name == "ModelEvaluator":
        from .evaluator import ModelEvaluator
        return ModelEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnomalyModel",
    "IsolationForestDetector",
    "VAEDetector",
    "ConflictPredictor",
    "ConflictLSTM",
    "BaselineModel",
    "ModelEvaluator",
]
