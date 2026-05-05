"""
Analysis Module
==============

Provides tools for analyzing AIS data and correlating with conflict events.

Classes:
    TrafficAnalyzer: Vessel traffic volume and density analysis
    BehavioralAnalyzer: Vessel behavior pattern analysis
    CorrelationAnalyzer: Statistical correlation with conflict events (Granger, DiD, ITS)
    DestinationAnalyzer: Port call and destination analysis
    ROTAnalyzer: Rate-of-Turn anomaly analysis
"""


def __getattr__(name):
    """Lazy import to avoid RuntimeWarning with python -m."""
    if name == "TrafficAnalyzer":
        from .traffic_analyzer import TrafficAnalyzer
        return TrafficAnalyzer
    elif name == "BehavioralAnalyzer":
        from .behavioral_analyzer import BehavioralAnalyzer
        return BehavioralAnalyzer
    elif name == "CorrelationAnalyzer":
        from .correlation_analyzer import CorrelationAnalyzer
        return CorrelationAnalyzer
    elif name == "DestinationAnalyzer":
        from .destination_analyzer import DestinationAnalyzer
        return DestinationAnalyzer
    elif name == "ROTAnalyzer":
        from .rot_analyzer import ROTAnalyzer
        return ROTAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TrafficAnalyzer",
    "BehavioralAnalyzer",
    "CorrelationAnalyzer",
    "DestinationAnalyzer",
    "ROTAnalyzer",
]
