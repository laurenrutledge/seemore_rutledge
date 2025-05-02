# Seemore VLM — Training Loop Redesign

## Overview: 
This project contains an extended and modular training framework for the **Seemore Vision-Language Model (VLM)**. 

The motivation of the re-design was to transform a basic training loop into a mature system capable of supporting: 
- Distributed training across multiple GPUs or nodes (while also allowing for the loop to be executed on a cpu if multiple GPUs / nodes are not available)
- Mixed Precision for time and storage efficiency 
- Configurable hyperparameters and logging
- Checkpointing and profiling for reproducibility and performance insight

-----



### Summary of Key Features Implemented: 
- **Configurable Training**
- **Distributed Training**
- **Automatic Mixed Precision (AMP)**
- **Checkpointing**
- **Loss Tracking & Visualization**
- **Profiling of Iteration Timing & Visualization**
- **Dataset Loader** to accomodate base64-encoded images and captions from CSV 
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
│   ├── vision_language_model.py   # Main Vision-Language Model (ViT + Decoder)
│   ├── decoder_language_model.py  # Decoder used in VLM (used, but not modified by Lauren)
│   ├── vision_transformer.py      # Vision Transformer encoder (used, but not modified)
│   ├── patch_embeddings.py        # Embeds image patches into vectors (used by ViT)
│   ├── block.py                   # Transformer block used in ViT
│   ├── attention.py               # Multi-head self-attention mechanism
│   └── multimodal_projector.py    # Projects image embeddings to decoder space
│
├── logs/                          # Output logs and visualizations
│   ├── loss_curve.png             # Training and validation loss plot
│   ├── timing_curve.png           # Forward, backward, and total iteration time plot
│   ├── losses_log.csv             # CSV of per-epoch train/val loss
│   └── timing_logs.csv            # CSV of per-iteration timing breakdown
│
├── checkpoints/                   # Saved model checkpoints (.pth files)
│
├── not_used_for_assignment/       # Archived notebooks/scripts not used in Lauren's training remodeling
│
├── train.py                       # CLI entry point to launch training with selected config
├── README.md                      # Project documentation
├── LICENSE                        # Project license file
└── .gitignore                     # Git ignore rules (e.g., logs, checkpoints, data)

```

---

## Methodology:
### Data Preprocessing: 
- Used USA Swimming records to filter and clean historical race data
- Removed non-relevant metadata to focus on performance-based attributes
- Converted swim times into a numerical scoring system based on USA Swimming time standards

### Exploratory Data Analysis (EDA):
- Visualized performance progression across different strokes and distances
- Analyzed variability and specialization trends in youth swim times

### Predictive Modeling:
- Baseline Model: ARIMA for time-series forecasting of swim times
- Machine Learning Model: Multi-output regression to predict event specialization

---

## Key Findings:
- ARIMA provides a reasonable statistical baseline but struggles with sprint events
- Multi-output regression improves predictive accuracy for short- and mid-distance races
- Long-distance events show more variability, requiring additional feature engineering

___

## How to Run the Code to Replicate Project 

### 1. Set Up the Environment
- First, activate the Conda environment and install the required dependencies:

```sh
conda create --name seemore_env python=3.10
conda activate seemore_env
pip install -r requirements.txt
```

### 2. Configure Parameters 
- Open the file: ./configs/default.yaml and ensure that you specify:
   - Desired Model Architecture (layers, heads, embedding sized)
   - Desired Training Parameters (num_epochs, learning_rate, batch_size)
   - Desired AMP, checkpointing, and logging preferences 

### 3. Begin Training! 
- Execute the training entry script using the configurations saved in default.yaml
```sh
python train.py --config configs/default.yaml
```
- Or, if there is a desire to override config options via CLI, execute: 
```sh
python train.py --config configs/default.yaml --device cuda --run_name debug_amp --log_wandb false
```

### 4. Run Exploratory Data Analysis (EDA)
- To analyze swim event performance as a function of age, run: 
```sh
jupyter notebook eda_mean_time_per_swim_event_vs_age_of_swimmer/calculating_mean_time_per_swim_event_as_function_of_age.ipynb
```

---
### Author:
Lauren Rutledge
April 2025


