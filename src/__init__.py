"""
Project Duration Estimation - Main Configuration

This module contains configuration settings for the project duration estimation system.
"""

from typing import Dict, Any
from omegaconf import OmegaConf
import os

# Default configuration
DEFAULT_CONFIG = {
    "data": {
        "synthetic": {
            "n_tasks": 10,
            "min_duration": 1,
            "max_duration": 10,
            "dependency_probability": 0.3,
            "seed": 42
        },
        "paths": {
            "raw": "data/raw/",
            "processed": "data/processed/",
            "synthetic": "data/synthetic/"
        }
    },
    "models": {
        "cpm": {
            "enabled": True,
            "weight_attribute": "duration"
        },
        "pert": {
            "enabled": True,
            "confidence_levels": [0.8, 0.9, 0.95],
            "simulation_runs": 10000
        },
        "monte_carlo": {
            "enabled": True,
            "n_simulations": 10000,
            "confidence_levels": [0.8, 0.9, 0.95]
        }
    },
    "evaluation": {
        "metrics": ["duration", "critical_path_length", "slack_time", "risk_score"],
        "cross_validation": {
            "enabled": False,
            "n_folds": 5
        }
    },
    "visualization": {
        "figure_size": [12, 8],
        "dpi": 300,
        "style": "seaborn-v0_8",
        "save_format": "png"
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "logs/project_duration.log"
    }
}

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from file or return default configuration.
    
    Args:
        config_path: Path to configuration file (YAML format)
        
    Returns:
        Configuration dictionary
    """
    if config_path and os.path.exists(config_path):
        return OmegaConf.load(config_path)
    return OmegaConf.create(DEFAULT_CONFIG)

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    OmegaConf.save(config, config_path)
