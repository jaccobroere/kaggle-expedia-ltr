# Kaggle: Expedia Hotel Searches, Learning-to-Rank (LTR)
This repository displays my solution to the [Personalize Expedia Hotel Searches - ICDM 2013](https://www.kaggle.com/competitions/expedia-personalized-sort/leaderboard?) competition held on Kaggle.

The solution can be divided into three main steps:
1. Data exploration
2. Data cleaning / imputation
3. Feature engineering
4. Hyperparameter optimization
5. Training phase and scoring

<!-- To install the `expedia_kaggle` package use run the following command from the root directory: -->

<!-- Windows:

    pip install -e .\modules\expedia_kaggle
Linux:

    pip install -e /modules/expedia_kaggle -->

## Downloading data
To download the data please install the `kaggle` package using `pip` and connect it with your Kaggle API token. Then run either

    ipython data/download_data.py

or

    sh data/download_data.sh



