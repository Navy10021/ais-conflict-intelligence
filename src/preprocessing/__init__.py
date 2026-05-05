"""
Data Preprocessing Module
=======================

Provides tools for loading, cleaning, feature engineering, and aggregating
AIS (Automatic Identification System) data from the Danish Maritime Authority.

Classes:
    AISDKLoader: Multi-day AIS data loader
    AISCleaner: Data cleaning and validation
    AISFeatureEngineer: Feature generation (6 categories)
    AISAggregator: Grid-cell and time-bucket aggregation
"""


def __getattr__(name):
    """Lazy import to avoid RuntimeWarning with python -m."""
    if name == "AISDKLoader":
        from .loader import AISDKLoader
        return AISDKLoader
    elif name == "AISCleaner":
        from .cleaner import AISCleaner
        return AISCleaner
    elif name == "AISFeatureEngineer":
        from .feature_engineer import AISFeatureEngineer
        return AISFeatureEngineer
    elif name == "AISAggregator":
        from .aggregator import AISAggregator
        return AISAggregator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AISDKLoader", "AISCleaner", "AISFeatureEngineer", "AISAggregator"]
