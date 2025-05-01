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
import torch
from training.distributed import (
    init_pytorch_distributed_mode,
    is_process_main_process,
    clean_up_distribution
)
from training.training_logger import (
    log_iteration_loss,
    log_epoch_ending,
    log_validation_loss
)
from training.visualization import (
    initialize_loss_tracking,
    log_losses_to_csv,
    plot_loss_curve,
    plot_timing_metrics
)
from training.profiling import (
    initialize_timing_log,
    initialize_timing_log_csv
)
from training.checkpointing import save_model_checkpoint
from training.helpers import (
    setup_device_and_model,
    setup_data_loaders,
    setup_optimizer_and_amp,
    training_step,
    validation_step
)



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

    is_distributed = config.get("distributed", False)

    # Step 1a: Initialize distributed mode (and get correct device!)
    if is_distributed:
        device = init_pytorch_distributed_mode(config)
    else:
        device = torch.device(config['device']) if torch.cuda.is_available() else torch.device('cpu')
        if is_process_main_process(config):
            print(f"Using single device setup on {device}")

    # Step 1b: Setup model (DDP-wrapped if needed)
    device, model = setup_device_and_model(config, is_distributed)

    # Step 2: Setup the dataloaders
    train_loader, val_loader = setup_data_loaders(config, is_distributed)

    # Step 3. Setup optimizer, loss function, AMP
    optimizer, criterion, scaler, use_amp = setup_optimizer_and_amp(config, model, device)

    # Step 4. Setup the logging and checkpointing
    if is_process_main_process(config):
        initialize_loss_tracking()
        initialize_timing_log()
        if config.get("save_checkpoints", False):
            os.makedirs(config.get("checkpoint_dir", "checkpoints"), exist_ok=True)


    # Step 5. Begin actual Training!
    global_step = 0
    model.train()
    best_val_loss = float("inf")

    # For loop over each epoch necessary (1 epoch = 1 run through dataset)
    for epoch in range(config["num_epochs"]):

        train_loss_total = 0.0

        for step, (images, captions) in enumerate(train_loader):
            loss, forward_time, backward_time, iteration_time = training_step(
                model, images, captions, optimizer, criterion, scaler, device, use_amp
            )
            train_loss_total += loss.item()

            # Log More steps in smaller datasets!
            if config["log_interval"] <= 1 or step % config["log_interval"] == 0:
                log_iteration_loss(global_step, loss.item())
                initialize_timing_log_csv(global_step, iteration_time, forward_time, backward_time)

            global_step += 1


        if is_process_main_process(config):
            log_epoch_ending(epoch)
            avg_val_loss = validation_step(model, val_loader, criterion, device)

            if is_distributed:
                # Synch validtion loss across processes
                val_loss_tensor = torch.tensor(avg_val_loss, device=device)
                torch.distributed.all_reduce(val_loss_tensor, op=torch.distributed.ReduceOp.SUM)
                avg_val_loss = val_loss_tensor.item() / config['world_size']


            avg_train_loss = train_loss_total / max(1, len(train_loader))

            log_validation_loss(global_step, avg_val_loss)
            log_losses_to_csv("logs/losses_log.csv", epoch, avg_train_loss, avg_val_loss)
            model.train()

            if config.get("save_checkpoints", False) and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_model_checkpoint(config, model, optimizer, best_val_loss, is_distributed, epoch)


    if is_process_main_process(config):
        plot_loss_curve("logs/losses_log.csv", "logs/loss_curve.png")
        plot_timing_metrics("logs/timing_logs.csv", "logs/timing_curve.png")
        print("Saved training curve to logs/loss_curve.png and the timing metrics to logs/timing_metrics.csv")


    if is_distributed:
        clean_up_distribution()