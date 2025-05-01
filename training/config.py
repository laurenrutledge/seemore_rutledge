"""
config.py

This file handles the configuration loading and parsing for the Seemore Vision Language Model
training sequence. Specifically, the file:
    1. Reads the configuration settings in from a YAML file
    2. Overrides any selected configuration values that are overridden via command-line arguments
    3. Prepares a "merged-config" dictionary (config values + command line args) to be used in training.

Author: Lauren Rutledge
Date: April 2025
"""


import yaml
import os
import argparse


def load_yaml_config(path: str) -> dict:
    """
    This function loads a configuration dictionary from a YAML file.
    
    Args:
        path: path to the YAML configuration file
        
    Returns:
        dict: the parsed configuration values
    """

    # Confirm the config file's path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    # If it exists, load it / read in config file
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config



def parse_args() -> argparse.Namespace:
    """
    This function parses the command-line arguments for configuration overrides.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    # Set up parser for reading in command line
    parser = argparse.ArgumentParser(description="Train a Vision Language Model")

    # Adding an argument for configuration file
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file (default: configs/default.yaml)")


    # These are the main override options (these may be added to in the future):
    # device
    parser.add_argument("--device", type=str,
                        help="Override device setting from config (e.g., 'cuda', 'cpu')")
    # run_name
    parser.add_argument("--run_name", type=str,
                        help="Optional override for the name of the training run")

    # log weights and biases
    parser.add_argument("--log_wandb", type=str, choices=["true", "false"],
                        help="Enable or disable Weights & Biases logging")

    return parser.parse_args()



def str_to_bool(value):
    """
    Helper function - converts string representations of booleans to actual bools,
    Leaves non-strings untouched.
    """
    if isinstance(value, str):
        return value.lower() == "true"
    return value



def merge_config(cli_args: argparse.Namespace, yaml_config: dict) -> dict:
    """
    This function merges the command-line argument with the overridden YAML config.

    Args:
        cli_args (argparse.Namespace): The command-line arguments that are to override YAML config
        yaml_config (dict): Configuration loaded from YAML file that may be overridden

    Returns:
        dict: Final merged configuration dictionary.
    """

    # Copy the YAML config to avoid modifying original!
    config = yaml_config.copy()

    # Apply CLI overrides ( if present)
    if cli_args.device:
        config["device"] = cli_args.device
    if cli_args.run_name:
        config["run_name"] = cli_args.run_name
    if cli_args.log_wandb: # string representations of booleans will convert to boolean:
        config["log_wandb"] = str_to_bool(cli_args.log_wandb)

    return config


