"""
train_model.py

This file contains the training loop logic that is executed when wanting to run the
Seemore Vision Language Model. The loop undergoes the following:
    1. The Model sets itself up using configs
    2. Data is loaded, training is initiated
    3. Logging of training loss occurs

Author: Lauren Rutledge
Date: April 2025
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler, random_split
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from modules.vision_language_model import VisionLanguageModel
from training.distributed import(
    init_pytorch_distributed_mode, is_process_main_process, clean_up_distribution)
from training.utils import CSVBase64ImageDataset



def train_model(config):
    """
    This function contains the main training loop for the vision-language model. It executes
     the following:
     1. The function selects a computing device to use for training (and/or uses distributed training)
     2. The function sets up the config that will be inputted into this model
     3. The function sets up the dataset that it will be ingesting
     4. The function sets up variables for the Optimizer, Loss, and AMP Setup
     5. The function executes the "training loop"!

    Args:
        config (dict): the config dictionary that contains the model and training settings!
    """

    # 1a. Initialize the Distributed Training (if it is enabled):
    is_distributed = config.get("distributed", False)
    if is_distributed:
        device = init_pytorch_distributed_mode(config)
    else:
    # 1b. If not enabled, select the compute device:
        device = torch.device(config['device']) if torch.cuda.is_available() else torch.device('cpu')
        if is_process_main_process(config):
            print(f"We are currently using a single device setup on the device: {device}")


    # 2. The function sets up the config / model instantiation - initialize model:
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


    # Wrap Model with DDP if using distributed training:
    if is_distributed:
        model = DDP(model,
                    device_ids=[config['local_rank']],
                    output_device=config['local_rank'])



    # 3. Load the real dataset (images/inputs.csv) from CSV with base64-encoded images
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(current_dir, "images", "inputs.csv")

    if is_process_main_process(config):
        print(f"Loading data from the csv file: {csv_path}")

    full_dataset = CSVBase64ImageDataset(csv_path, config["img_size"])

    train_val_ratio = config.get("train_val_split", 0.8)
    train_size = int(train_val_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])



    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=config['world_size'], rank=config['rank'])
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config["batch_size"])
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config["batch_size"])
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])



    # 4. Set up the optimizer, loss, and AMP:
    optimizer = optim.AdamW(model.parameters(),
                            lr=float(config["learning_rate"]),
                            weight_decay=float(config["weight_decay"]))

    criterion = nn.CrossEntropyLoss()

    use_amp = config.get("amp", False) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)


    # 5. Begin Training!
    model.train()
    for epoch in range(config["num_epochs"]):
        for step, (images, captions) in enumerate(train_loader):
            images, captions = images.to(device), captions.to(device)

            optimizer.zero_grad()

            # Make sure autocast can handle "cuda" AND "cpu" - dependent on which device currently being used
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images, captions[:, :-1])  # decoder input = captions without last token

                # Make sure the input is now used to predict the NEXT token:
                # Align logits and targets to the same length (20 → 19)
                targets = captions[:, 1:]  # targets = next tokens

                print("logits shape:", logits.shape)
                print("targets shape:", targets.shape)

                # Align predictions to targets (cut off last logit step)
                logits = logits[:, :targets.size(1), :]

                # Flatten both for loss
                loss = criterion(
                    logits.contiguous().view(-1, logits.size(-1)),  # → [3*19, 100]
                    targets.contiguous().view(-1)  # → [3*19]
                )


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % config["log_interval"] == 0 and is_process_main_process(config):
                print(f"[Epoch {epoch}] Step {step} | Loss {loss.item():.4f}")

        if is_process_main_process(config):
            print(f"End of epoch {epoch} \n")

            model.eval()
            val_loss_total = 0.0

            with torch.no_grad():
                for val_images, val_captions in val_loader:
                    val_images, val_captions = val_images.to(device), val_captions.to(device)
                    val_logits = model(val_images, val_captions[:, :-1])
                    val_logits = val_logits[:, :-1, :]  # match targets
                    val_targets = val_captions[:, 1:]
                    val_loss = criterion(val_logits.reshape(-1, val_logits.size(-1)), val_targets.reshape(-1))
                    val_loss_total += val_loss.item()

            avg_val_loss = val_loss_total / len(val_loader)
            print(f"[Validation] Average Loss after Epoch {epoch}: {avg_val_loss:.4f}")
            model.train()

    if is_distributed:
        clean_up_distribution()