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
 seemore/
├── configs/
│   └── default.yaml
├── training/
│   ├── train_model.py
│   ├── helpers.py
│   ├── config.py
│   ├── checkpointing.py
│   ├── profiling.py
│   ├── training_logger.py
│   ├── utils.py
├── run_training.py
├── images/
│   └── inputs.csv
├── logs/
│   ├── losses_log.csv
│   ├── timing_logs.csv
│   └── *.png
└── checkpoints/
    └── *.pth

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
conda create --name cs229_project_env python=3.10
conda activate cs229_project_env
pip install -r requirements.txt
```

### 2. Obtain the Data
- Ensure you have the cleaned dataset ready:
- Use swimmers_cleaned.csv located in the usaa_swim_data/ directory

### 3. Run Feature Engineering
- Execute the feature engineering script to generate additional features:
```sh
jupyter notebook feature_engineering/adding_features_to_dataset.ipynb
```
### 4. Run Exploratory Data Analysis (EDA)
- To analyze swim event performance as a function of age, run: 
```sh
jupyter notebook eda_mean_time_per_swim_event_vs_age_of_swimmer/calculating_mean_time_per_swim_event_as_function_of_age.ipynb
```
### 5. Run ARIMA Model
- To compute ARIMA-based predictions for event specialization, execute:
```sh
jupyter notebook arima/calculate_mean_time_per_event_specialized_vs_non_specialized.ipynb
```
### 6. Run Multi-Output Regression
- To train and evaluate the multi-output regression model:
```sh
jupyter notebook multi_output_regression/multi_output_regression_v2.ipynb
```




