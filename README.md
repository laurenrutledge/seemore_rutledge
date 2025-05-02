# Seemore VLM — Training Loop Redesign

## Overview: 
This project contains an extended and modular training framework for the **Seemore Vision-Language Model (VLM)**. 

The motivation of the re-design was to transform a basic training loop into a mature system capable of supporting: 
- Distributed training across multiple GPUs or nodes (while also allowing for the loop to be executed on a cpu if multiple GPUs / nodes are not available)
- Automatic Mixed Precision (AMP) for improved time and storage efficiency
- YAML-configurable Hyperparameters that also allows for CLI overrides 
- Loss Logging, Profiling, and Visualization
- Checkpointing and profiling for reproducibility and validation performance

-----

While working on the assignment, the following features were focused upon: 

**Note: All key additions to the training loop (implemented by Lauren as part of this assignment) are located in the ./training/ directory** 

### Summary of Key Features Implemented: 
- **Configurable Training** via the 'default.yaml' file, while also allowing for CLI overrides 
- **Distributed Training** via Pytorch's Distributed Data Parallel (DDP)
- **Automatic Mixed Precision (AMP) Support** to allow for faster GPU training
- **Checkpointing** of the best-performing models 
- **Loss Tracking & Visualization** to confirm that loss functions were working as expected 
- **Profiling of Iteration Timing & Visualization**, outputted in CSV and PNG formats
- **Dataset Loader** to accomodate base64-encoded images and captions in CSV format  
-----

## Repository Structure: 
```plaintext
seemore_rutledge_1/
├── configs/
│   └── default.yaml               # YAML configuration for training parameters and model architecture
│
├── training/                      # Core training logic and utilities
│   ├── train_model.py             # Main training loop implementation
│   ├── helpers.py                 # Model, optimizer, dataloader, and AMP setup
│   ├── config.py                  # YAML + CLI configuration merging logic
│   ├── checkpointing.py           # Model checkpoint saving based on validation loss
│   ├── profiling.py               # Timing utilities and plotting for forward/backward/total time
│   ├── training_logger.py         # Iteration and epoch logging functions
│   └── utils.py                   # Custom Dataset loader for base64-encoded image-caption CSV
│
├── modules/                       # Model architecture components
│   ├── vision_language_model.py   # Main Vision-Language Model (ViT + Decoder) (used, but not modified by Lauren)
│   ├── decoder_language_model.py  # Decoder used in VLM (used, but not modified by Lauren)
│   ├── vision_transformer.py      # Vision Transformer encoder (used, but not modified by Lauren)
│   ├── patch_embeddings.py        # Embeds image patches into vectors (used by ViT) (used, but not modified by Lauren)
│   ├── block.py                   # Transformer block used in ViT (used, but not modified by Lauren)
│   ├── attention.py               # Multi-head self-attention mechanism (used, but not modified by Lauren)
│   └── multimodal_projector.py    # Projects image embeddings to decoder space (used, but not modified by Lauren)
│
├── logs/                          # Output logs and visualizations
│   ├── loss_curve.png             # Training and validation loss plot
│   ├── timing_curve.png           # Forward, backward, and total iteration time plot
│   ├── losses_log.csv             # CSV of per-epoch train/val loss
│   └── timing_logs.csv            # CSV of per-iteration timing breakdown
│
├── checkpoints/                   # Saved model checkpoints (.pth files)
│
├── not_used_for_assignment/       # Archived notebooks/scripts not used or even referenced in Lauren's training remodeling
│
├── train.py                       # CLI entry point to launch training with selected config
├── README.md                      # Project documentation
├── LICENSE                        # Project license file
└── .gitignore                     # Git ignore rules (e.g., logs, checkpoints, data)

```
___

## How to Run the Code Lauren's Updated Training Loop 

### 1. Set Up the Environment
- First, activate the Conda environment and install the required dependencies:

```sh
conda create --name seemore_env python=3.10
conda activate seemore_env
pip install -r requirements.txt
```
### 2. Prepare Input Data
- Ensure that you have a base64-encoded image-caption dataset saved as:
     ```sh
   ./images/inputs.csv
      ```
- In the file, each row should include:
   - `b64string_images`: the base64-encoded image
   - `caption`: the associated text caption

If this file is missing or incorrectly formatted, training will not start.

- Note: With more time, support for additional input formats (e.g., raw image folders or JSON-based datasets) could be added as a next step! 

  
### 3. Configure Parameters 
- Open the file: ./configs/default.yaml and specify:
   - Desired Model Architecture (layers, heads, embedding sizes)
   - Desired Training Parameters (num_epochs, learning_rate, batch_size)
   - Desired AMP, checkpointing, and logging preferences 

### 4. Begin Training! 
- Execute the training entry script using the configurations saved in default.yaml
```sh
python train.py --config configs/default.yaml
```
- Or, if there is a desire to override config options via CLI, execute: 
```sh
python train.py --config configs/default.yaml --device cuda --run_name debug_amp --log_wandb false
```


___

## Output Artifacts:
- checkpoints/: Saved models based on best val loss
- logs/losses_log.csv: Raw loss values per epoch
- logs/loss_curve.png: Loss curve plot
- logs/timing_logs.csv: Timing breakdown per iteration
- logs/timing_curve.png: Visual timing plot (iteration/forward/backward)

___
## Final Notes: 
- Files in the `not_used_for_assignment/` directory were preserved for reference but were not used in Lauren’s redesigned training loop.
- This implementation assumes availability of PyTorch and (optionally) GPUs.
  
---
### Author:
Lauren Rutledge   
MS Candidate; Mathematical & Computational Engineering  
April 2025  



