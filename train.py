
"""
train.py

This file contains the command-line interface "entry point" to begin training for the Seemore
Vision Language Model. The script will parse command-line arguments, then load configs from the
YAML file, and then follow the "training loop" described in ./training/train_model.py. This file
was created to allow for flexibility, scriptability, and reproducibility to hopefully make it useful
for testing and debugging if used again.

Author: Lauren Rutledge
Date: April 2025
"""

from training.config import load_yaml_config, parse_args, merge_config
from training.train_model import train_model

def main():

    # Step 1: Parse command-line arguments
    cli_args = parse_args()

    # Step 2: Load the YAML config file
    yaml_config = load_yaml_config(cli_args.config)

    # Step 3: Merge the CLI inputs that should override the YAML config file
    config = merge_config(cli_args, yaml_config)

    # Step 4: Launch the training portion with the merged configs
    train_model(config)


if __name__ == "__main__":
    main()