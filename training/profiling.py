"""
profiling.py

This file contains the profiles of the forward pass of the Seemore Vision Language Model using
Pytorch's built-in profiler. Note - profiling measures how much time and memory different parts
of this code take, and can help with seeing whether DDP or multi-GPU is actually helping our time /
space optimization. It also helps with ensuring we are using the correct batch size and with avoiding
any CPU or GPU stalls in our process.

The outputs of this file includes CPU and GPU time for the key operations.

profiling.py is different from  training_logger.pyin that it records performance metrics like:
- how long each iteration took
- how long forward and backward passes took

Author: Lauren Rutledge
Date: April 2025
"""


import time
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt

# Call upon a timing_tracking_log path to save timing stats calculated below:
_timing_log_path = os.path.join("logs", "timing_logs.csv")


def initialize_timing_log():
    """
    This function initializes a csv file that will be used for tracking the timing metrics
    per iteration.
    """

    os.makedirs("logs", exist_ok=True)

    with open(_timing_log_path, mode="w", newline="") as timing_log:
        writer = csv.writer(timing_log)
        writer.writerow(["iteration", "iteration_time", "forward_time", "backward_time"])


def initialize_timing_log_csv(iteration, iteration_time, forward_time, backward_time):
    """
    This function logs the timing of one iteration to a new line of the csv file: timing_log_csv

    Args:
        iteration (int): The current iteration number.
        iteration_time (float): The current iteration time.
        forward_time (float): The current forward time.
        backward_time (float): The current backward time.
    """

    with open(_timing_log_path, mode="a", newline="") as timing_log:
        writer = csv.writer(timing_log)
        writer.writerow([iteration, iteration_time, forward_time, backward_time])



def start_timer():
    """
    This function starts a high-precision timer!
    """
    return time.perf_counter()


def end_timer(start):
    """
    This function ends the high-precision timer!
    """
    return time.perf_counter() - start


def plot_timing_metrics(csv_path="logs/timing_logs.csv", output_path="logs/timing_plot.png"):
    """
    This function plots the timing metrics (iteration, forward, backward) from the timing CSV.

    Args:
        csv_path (str): Path to the CSV file containing timing logs.
        output_path (str): Path to save the generated plot image.
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Timing log not found: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df["iteration"], df["iteration_time"], label="Total Iteration Time")
    plt.plot(df["iteration"], df["forward_time"], label="Forward Pass Time")
    plt.plot(df["iteration"], df["backward_time"], label="Backward Pass Time")
    plt.xlabel("Iteration")
    plt.ylabel("Time (seconds)")
    plt.title("Training Profiling: Time per Iteration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved profiling plot to {output_path}")