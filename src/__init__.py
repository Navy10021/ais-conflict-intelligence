"""
Maritime Conflict Intelligence System (MCIS)
===========================================

A comprehensive system for detecting maritime behavioral anomalies
correlated with armed conflicts using AIS data.

Version: 0.1.0
Author: Your Name
Email: your.email@example.com
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from pathlib import Path

# Package root directory
ROOT_DIR = Path(__file__).parent.parent


def load_config(config_path: str = None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = ROOT_DIR / "config" / "settings.yaml"
    try:
        import yaml
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except ImportError:
        import json
        # Fallback to json if yaml not available
        with open(config_path.replace(".yaml", ".json"), "r") as f:
            return json.load(f)


__all__ = ["__version__", "__author__", "__email__", "load_config", "ROOT_DIR"]
