import yaml
import json
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Config dictionary
    """
    path = Path(config_path)
    
    if path.suffix in ['.yaml', '.yml']:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
    elif path.suffix == '.json':
        with open(path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")
        
    return config


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model configuration parameters."""
    return config.get('model', {})


def get_optimizer_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract optimizer configuration parameters."""
    return config.get('optimizer', {})


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract training configuration parameters."""
    return config.get('training', {})


def get_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract data configuration parameters."""
    return config.get('data', {})
