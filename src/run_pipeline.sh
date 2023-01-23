#!/bin/bash
# Install requirements.txt
python -m pip install -r requirements.txt

# Run the pipeline
python 01_subset_data.py
python 02_imputation_feature_engineering.py
python 03_hyperparameter_tuning.py
python 04_model_fit_score.py