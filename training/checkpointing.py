"""
checkpointing.py

This module defines the checkpointing utility used during training of the Seemore Vision-Language Model.
It saves the model's state_dict when validation loss improves, and supports both single-device and
DistributedDataParallel (DDP) training modes.

Author: Lauren Rutledge
Date: April 2025
"""


import os
import torch

def save_model_checkpoint(config, model, optimizer, val_loss, is_distributed=False, epoch=0):

    """
    This function saves a checkpoint of the model if validation loss improves!

    Args:
        model: PyTorch model or DDP-wrapped model
        is_distributed (bool): Whether DDP is being used
        epoch (int): Current epoch
        val_loss (float): Validation loss to include in filename
        config (dict): Configuration dictionary
    """
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    run_name = config.get("run_name", "seemore")
    filename = f"{run_name}_epoch{epoch}_val{val_loss:.4f}.pth"
    path = os.path.join(checkpoint_dir, filename)

    torch.save(model.module.state_dict() if is_distributed else model.state_dict(), path)
    print(f"✓ Saved improved checkpoint to {path}")
