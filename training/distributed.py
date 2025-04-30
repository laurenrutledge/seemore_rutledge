"""
distributed.py

This file contains utility functions for setting up PyTorch Distributed Data Parallel (DDP) training
for the Seemore Vision Language Model.

The functions in the file do the following:
    1. Initialize Pytorch Distributed Data Parallel via torch.distributed
    2. Figures out whether the current process is the rank 0 process (the main process)
    3. The last function cleans up the process group after training to free up resources

Author: Lauren Rutledge
Date: April 2025
"""

import os
import torch
import torch.distributed as dist


def init_pytorch_distributed_mode(config):
    """
    This function is initializing the Pytorch Distributed Data Parallel (DDP) mode if the
    environmental variables were correctly set. If so, the function sets up the proper device,
    initializes the process group, and then updates the inputted config with the current process's
    rank and world size

    Args:
        config (dict): the training configuration dictionary

    Returns:
        torch.device: The local device used for this process:
    """

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        config['rank'] = int(os.environ['RANK'])
        config['world_size'] = int(os.environ['WORLD_SIZE'])
        config['local_rank'] = int(os.environ.get('LOCAL_RANK', 0))
    else:
        print("We are currently NOT using distributed training")
        config['rank'] = 0
        config['world_size'] = 1
        config['local_rank'] = 0


    # Now, we want to set the correct device to use:
    torch.cuda.set_device(config['local_rank'])
    device = torch.device("cuda", config['local_rank'])

    # Initializing the Distributed Process Group now:
    dist.init_process_group(backend='nccl', init_method='env://')

    # Use dist.barrier() to ensure all processes are being synched
    dist.barrier()


    if config['rank'] == 0:
        print("Distributed training is being initialized.")
    return device



def is_process_main_process(config):
    """
    This function confirms whether the process that is occurring is the main one
    (rank 0 process), which is important to know because only the main process should be
    writing the logs (we do not want duplicate logging).

    Args:
        config (dict): the training configuration dictionary

    Returns:
        bool: True if the process that is occurring is the main one
    """
    return config.get('rank', 0) == 0



def clean_up_distribution():
    """
    This function cleans up what was the distributed training environment, and 
    should be called after training is complete to free up the resources! 
    """

    dist.destroy_process_group()