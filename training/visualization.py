"""
visualization.py

This file contains functions that allow for storing of the training and validation loss data
in a csv, in addition to a function that will create a training vs. validation loss plot for the
Seemore Vision Language Model.

The Outputs of the losses / plots are stored to a csv and are plotted using the matplotlib library


Author: Lauren Rutledge
Date: April 2025

"""
import csv
import os
import matplotlib.pyplot as plt

def initialize_loss_tracking(log_dir="logs"):
    """
    This function initializes a logs directory and creates a csv log file for the logs to be stored

    Args:
        log_dir (str): The directory to store the logs / csv in
    """

    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "losses_log.csv")

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "train_loss", "val_loss"])

    return csv_path


def log_losses_to_csv(csv_path, iteration, train_loss, val_loss):
    """
    This function takes the training and validation losses and appends them to a new row
    within the csv log file.

    Args:
        csv_path (str): The path to the csv log file
        iteration (int): The global training iteration number the losses are associated with
        train_loss (float): The training loss value at the recorded step
        val_loss (float): The validation loss value after each epoch
    """

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([iteration, train_loss, val_loss])


def plot_loss_curve(csv_path, output_dir="logs"):
    """
    This function reads in the losses log and plots the training vs. validation loss curve.

    Args:
        csv_path (str): The path to the csv log file
        output_dir (str): The directory to store the plotted loss curve
    """

    iterations = []
    train_losses = []
    val_losses = []

    with open(csv_path, mode="r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            iterations.append(int(row["iteration"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    plt.figure(figsize=(10, 5))
    plt.plot(iterations, train_losses, label="Training Loss")
    plt.plot(iterations, val_losses, label="Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(output_path)
    plt.close()

