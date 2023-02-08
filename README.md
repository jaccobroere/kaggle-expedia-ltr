# Kaggle: Expedia Hotel Searches, Learning-to-Rank (LTR)
This repository displays my solution to the [Personalize Expedia Hotel Searches - ICDM 2013](https://www.kaggle.com/competitions/expedia-personalized-sort/leaderboard?) competition held on Kaggle.

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

## Downloading data
To download the data please install the `kaggle` package using `pip` and connect it with your Kaggle API token. Then run either

    ipython data/download_data.py

or

    sh data/download_data.sh
    
Alternatively, you can download the data from the competition on the Kaggle website directly


## Step 1: Subsetting the data
The data for this competition is quite large. Therefore, to reduce the running time of the data preprocessing and training later on, we can take a subset of the training data for our train and validation sets. The data is quite uniform, so this should not harm perfomance too harshly. However, if you have the computing power, you can change the subset sizes in `config.ini`. 


