"""
training_logger.py

This file contains the logging functions for the Seemore Vision Language Model training process.
Specifically, it contains the print statements that are called upon at the end of each epoch /
iteration during the training. The goal of this file was to ensure that model training, validations,
and performance statistics can be consistently tracked and / or referenced throughout the process.

training_logger.py is different from profiling.py in that it records semantic training progress like:
- loss at iteration
- validation loss per epoch
- epoch completion markers

Author: Lauren Rutledge
Date: April 2025
"""


def log_iteration_loss(global_iteration, loss):
    """
    This function is responsible for printing the training loss at the end of each
    ITERATION, or a single parameter update step (one batch has been processed).

    Args:
        global_iteration (int): The current iteration number across all epochs
        loss (float): The loss value that is to be reported at the end of each epoch
    """

    print(f"Loss at iteration {global_iteration}: {loss:.4f}")



def log_epoch_ending(epoch):
    """
    This function will print a message at the end of each training epoch.

    Args:
        epoch (int): The current epoch number / index
    """

    print(f"At end of epoch: {epoch}\n")


def log_validation_loss(epoch, average_val_loss):
    """
    This function is responsible for printing the validation loss at the end of an epoch

    Args:
        epoch (int): The current epoch number / index
        average_val_loss (float): The average loss on the validation set.
    """

    print(f"Validation Loss after epoch {epoch}: {average_val_loss:.4f}")
