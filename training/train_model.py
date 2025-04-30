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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from modules.vision_language_model import VisionLanguageModel
from training.distributed import(
    init_pytorch_distributed_mode, is_process_main_process, clean_up_distribution)




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


    # Wrap Model with DDP if using distributed training:
    if is_distributed:
        model = DDP(model, device_ids=[config['local_rank']], output_device=config['local_rank'])

    # Delete below for actual implementation -:

    dummy_images = torch.randn(100, 3, config['img_size'], config['img_size'])
    dummy_labels = torch.randint(0, config['vocab_size'], (100, 20))

    dataset = torch.utils.data.TensorDataset(dummy_images, dummy_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'])

    # 4. Set up optimizer, loss, AMP

    optimizer = optim.AdamW(model.parameters(),
                            lr=float(config['learning_rate']),
                            weight_decay=float(config['weight_decay']))
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=config.get("use_amp", False))

    # 5. Run the training loop
    model.train()
    for epoch in range(config['num_epochs']):
        for step, (images, targets) in enumerate(dataloader):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=config.get("use_amp", False)):
                logits = model(images, targets[:, :-1])
                T = min(logits.size(1), targets[:, 1:].size(1))  # align sequence length
                loss = criterion(logits[:, :T].reshape(-1, logits.size(-1)), targets[:, 1:1 + T].reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            #if step % config['log_interval'] == 0 and is_process_main_process(config):
                #print(f"[Epoch {epoch}] Step {step} | Loss: {loss.item():.4f}")

        if is_process_main_process(config):
            print(f"Validation (not implemented) | End of Epoch {epoch}\n")

    # Clean up distributed resources if needed
    if config.get("distributed", False):
        clean_up_distribution()