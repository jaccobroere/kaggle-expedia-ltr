#!/bin/bash
# Run the pipeline
python src/01_subset_data.py
python src/02_imputation_feature_engineering.py
python src/03_hyperparameter_optimization.py
# python src/04_model_fit_score.py