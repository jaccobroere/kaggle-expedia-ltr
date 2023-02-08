# Kaggle: Expedia Hotel Searches, Learning-to-Rank (LTR)
This repository displays my solution to the [Personalize Expedia Hotel Searches - ICDM 2013](https://www.kaggle.com/competitions/expedia-personalized-sort/leaderboard?) competition held on Kaggle. The final NDCG@38 scores were 0.51004 and 0.51152 and on the private and public leaderboard, respectively, using 10 percent of the training data.

## Problem statement and evaluation method
The problem statement can be read in [problem_statement.pdf](problem_statement.pdf). Due to LaTeX rendering issues with GitHub's Markdown.

## Solution overview
The solution can be divided into the following main steps:
1. Data subsetting
2. Feature engineering and missing value imputation
3. Hyperparameter optimization
4. Training phase and scoring

<!-- To install the `expedia_kaggle` package use run the following command from the root directory: -->

<!-- Windows:

    pip install -e .\modules\expedia_kaggle
Linux:

    pip install -e /modules/expedia_kaggle -->

### Step 0: Downloading data
To download the data please install the `kaggle` package using `pip` and connect it with your Kaggle API token. Then run either

    ipython data/download_data.py

or

    sh data/download_data.sh
    
Alternatively, you can download the data from the competition on the Kaggle website directly


### Step 1: Subsetting the data
The data for this competition is quite large. Therefore, to reduce the running time of the data preprocessing and training later on, we can take a subset of the training data for our train and validation sets. The data is quite uniform, so this should not harm perfomance too harshly. However, if you have the computing power, you can change the subset sizes in `config.ini`. The data subsetting is performed in `src/01_subset_data.py`.

### Step 2: Feature engineering and missing value imputation
The data is quite messy, so we need to do some feature engineering and missing value imputation. The feature engineering and missing value imputation is performed together in `src/02_imputation_feature_engineering.py`.

### Step 3: Hyperparameter optimization
To optimize the hyperparameters of the models discussed in the previous section, we employ the open-source hyperparameter optimization framework `optuna`. `optuna` addresses problems inherent to common hyperparameter optimization approaches, such as the requirement of a static search space for the parameters, through employment of dynamically constructed parameter spaces. Moreover, `optuna` improves on the cost-effectiveness of the optimization framework by implementing efficient sampling and pruning mechanisms. Our usage of `optuna` employs a Tree-structured Parzen Estimator, which fits a pair of Gaussian Mixture Models (GMM). Namely, one GMM is fitted to the set of parameters that are associated with the best evaluation score, in our case the average NDCG@38 on the validation set as discussed in section, while the other GMM is fitted on the remaining parameter values to be tried. For efficient pruning, `optuna` utilizes Asynchronous Successive Halving (ASHA), which is more aggressive than regular Successive Halving, with more cost-effective hyperparameter optimization as a result. The hyperparameter optimization is performed in `src/03_hyperparameter_optimization.py`.

### Step 4: Training phase and scoring
The training phase and scoring is performed in `src/03_model_fit_score.py`. The training is performed using the `lightgbm` package, which is a gradient boosting framework that uses tree based learning algorithms. The `lightgbm` package is a fast, distributed, high performance gradient boosting framework based on decision tree algorithms, used for ranking, classification and many other machine learning tasks. The `lightgbm` package is used to train the models discussed in the previous section. The scoring is performed on the test set, and the final evaluation score is calculated as the average NDCG@38 score over all queries $q_i \in \mathcal{Q}$ in the test set.



