"""
train_model.py

This file contains the training loop that is executed when wanting to run the
Seemore Vision Language Model. The loop undergoes the following:
    1. The Model sets itself up using configs
    2. Data is loaded
    3. Logging of training loss occurs

Author: Lauren Rutledge
Date: April 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from modules.vision_language_model import VisionLanguageModel


def train_model(config):
    """
    This function contains the main training loop for the vision-language model. It executes
     the following:
     1. The function selects a computing device to use for training
     2. The function sets up the config that will be inputted into this model
     3. The function sets up the dataset that it will be ingesting
     4. The function sets up variables for the Optimizer, Loss, and AMP Setup
     5. The function executes the "training loop"!

    Args:
        config (dict): the config dictionary that contains the model and training settings!
    """

    # 1. Select the Compute Device:
    if torch.cuda.is_available():
        device = torch.device(config['device'])
    else:
        device = torch.device('cpu')
    print(f"Using computing device: {device}")

    # 2. The function sets up the config / model instantiation:
    model = VisionLanguageModel(n_embd=config['n_embd'],
                                image_embed_dim=config['image_embed_dim'],
                                vocab_size=config['vocab_size'],
                                n_layer=config['n_layer'],
                                img_size=config['img_size'],
                                patch_size=config['patch_size'],
                                num_heads=config['n_head'],
                                num_blks=config['num_blks'],
                                emb_dropout=config['dropout'],
                                blk_dropout=config['dropout']).to(device)



