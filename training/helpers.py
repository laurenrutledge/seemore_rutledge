"""
helpers.py

This file contains all of the helper functions referenced in train_model.py, which is the
main training loop for the Seemore Vision-Language Model. These include the helpers for:
- Setting up the model
- Loading the data into the model
- Managing Optimization and AMP
- Executing the Training Steps
- Evaluating the Model's Validation Losses

Author: Lauren Rutledge
Date: April 2025
"""

import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, DistributedSampler


from modules.vision_language_model import VisionLanguageModel
from training.utils import CSVBase64ImageDataset
from training.profiling import start_timer, end_timer



def setup_device_and_model(config, is_distributed):
    """
    This function sets up the compute device that will be used for training / testing
    (CPU or GPU) and initializes the model: Vision-language-model that our implementation uses.
    If distribution_training is enabled, this function also wraps the model with DistributedDataParallel
    (DDP).

    Args:
        config (dict): The configuration dictionary.
        is_distributed (bool): Whether or not we are using distributed training.
    """

    # Prefer CUDA if available, fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The function sets up the config / model instantiation - initialize model:
    model = VisionLanguageModel(
        n_embd=config["n_embd"],
        image_embed_dim=config["image_embed_dim"],
        vocab_size=config["vocab_size"],
        n_layer=config["n_layer"],
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        num_heads=config["n_head"],
        num_blks=config["num_blks"],
        emb_dropout=config["dropout"],
        blk_dropout=config["dropout"]
    ).to(device)

    # Wrap Model with DDP if using distributed training:
    if is_distributed:
        model = DDP(model, device_ids=[config['local_rank']], output_device=config['local_rank'])

    return device, model


def setup_data_loaders(config, is_distributed):
    """
    This function loads in the dataset (inputs.csv), splits it into the training and validation sets
    (which is an 80-20 split), and then returns the DataLoader Objects for the two sets.
    DistributedSampler is also used if distributed training is enabled.

    Args:
        config (dict): The configuration dictionary.
        is_distributed (bool): Whether or not we are using distributed training (DDP).
    """

    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(current_dir, 'images', 'inputs.csv')
    full_dataset = CSVBase64ImageDataset(csv_path, config["img_size"])

    train_val_ratio = config.get("train_val_split", 0.8)
    train_size = int(train_val_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=config['world_size'], rank=config['rank'])
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config["batch_size"])
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config["batch_size"])
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    return train_loader, val_loader


def setup_optimizer_and_amp(config, model, device):
    """
    This function sets up the optimizer (AdamW), the loss criterion (Cross Entropy), and
    automatic mixed precision (amp) scaler that are used in training.

    Args:
        config (dict): The configuration dictionary.
        model (VisionLanguageModel, or nn.Module): The model whose parameters we are optimizing in training
        device (torch.device): The device used for trainin g
    """

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"])
    )

    criterion = nn.CrossEntropyLoss()
    use_amp = config.get("amp", False) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    return optimizer, criterion, scaler, use_amp


def training_step(model, images, captions, optimizer, criterion, scaler, device, use_amp):
    """
    This function executes one single iteration of the training, which includes:
        1. Moving the data to the device
        2. Performs a forward pass (this occurs with AMP if it is enabled)
        3. The function computes/performs the loss function
        4. The Function performs the backward pass and optimizer step
        5. The function returns the timing statistic

    Args:
        model (VisionLanguageModel): The model whose parameters we are optimizing in training / being trained
        images (torch.Tensor): The input images into our model
        captions (torch.Tensor): The input captions into our model - the Ground Truth token sequences
        optimizer (torch.optim.Optimizer): The optimizer whose parameters we are optimizing in training
        criterion (torch.nn.CrossEntropyLoss): The loss function
        scaler (GradScaler): A scaler that scales the loss function / AMP gradient scaler
        device (torch.device): The device used for training / computation
        use_amp (bool): Whether or not we are using AMP
    """

    iteration_start = start_timer()
    images, captions = images.to(device), captions.to(device)

    optimizer.zero_grad()

    forward_start = start_timer()
    with autocast(device_type=device.type, enabled=use_amp):
        # First, we shift the input images / captions for the decoder to properly read
        logits = model(images, captions[:, :-1])
        # Create target variable to be able to predict next token
        targets = captions[:, 1:]
        # Now, we align time dimensions
        logits = logits[:, :targets.size(1), :]
        loss = criterion(
            logits.contiguous().view(-1, logits.size(-1)),
            targets.contiguous().view(-1)
        )
    forward_time = end_timer(forward_start)

    backward_start = start_timer()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    backward_time = end_timer(backward_start)

    iteration_time = end_timer(iteration_start)
    return loss, forward_time, backward_time, iteration_time


def validation_step(model, val_loader, criterion, device):
    """
    This function computes the average validation loss over all batches in the validation loader.

    Args:
        model (nn.Module): The model whose parameters we are optimizing in training / being trained
        val_loader (torch.utils.data.DataLoader): The validation data loader
        criterion (torch.nn.CrossEntropyLoss): The loss function
        device (torch.device): The device used for training / computation
    """

    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for val_images, val_captions in val_loader:
            val_images, val_captions = val_images.to(device), val_captions.to(device)

            # Get input captions (all but the last token)

            val_logits = model(val_images, val_captions[:, :-1])
            val_logits = val_logits[:, :-1, :]
            val_targets = val_captions[:, 1:]
            val_loss = criterion(val_logits.reshape(-1, val_logits.size(-1)), val_targets.reshape(-1))
            total_val_loss += val_loss.item()

    model.train()

    return total_val_loss / max(1, len(val_loader))