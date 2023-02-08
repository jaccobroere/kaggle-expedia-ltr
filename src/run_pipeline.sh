#!/bin/bash
# Run the pipeline
echo "Running pipeline"
echo "Step 1: Subset data"
python src/01_subset_data.py

echo "Step 2: Imputation and feature engineering"
python src/02_imputation_feature_engineering.py

echo "Step 3: Hyperparameter optimization"
python src/03_hyperparameter_optimization.py

echo "Step 4: Model fit and score"
python src/04_model_fit_score.py