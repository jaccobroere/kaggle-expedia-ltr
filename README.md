# Kaggle: Expedia Hotel Searches, Learning-to-Rank (LTR)
This repository displays my solution to the [Personalize Expedia Hotel Searches - ICDM 2013](https://www.kaggle.com/competitions/expedia-personalized-sort/leaderboard?) competition held on Kaggle. The final NDCG@38 scores were xxx and xxx on the public and private leaderboard, respectively, using 10 percent of the training data.

## Problem statement and evaluation method
The learning to rank problem comprises data on a set of queries, $\mathcal{Q}$, denote $\bm{q}_i \in \mathcal{Q}$ as the $i$th search query in the dataset with its corresponding features. Each query, $\bm{q}_i$, has a set of properties (hotels), $\mathcal{P}_i = \{p_{i1},\ldots ,p_{in_i}\}$, associated with it. Note that the number of hotels associated with each query need not be the same, i.e., each $\bm{q}_i$ has $n_i$ properties associated with it. Every property $p_{ij} \in \mathcal{P}_i$ has corresponding a true relevance score $r_{ij} \in \bm{r}_i$. In our problem, the true relevance score for a certain hotel in a query is equal to 5 if the hotel was booked, equal to 1 if it was clicked on, and zero otherwise. Now, the model we want to build has the following structure:
$$
    M: \bm{q}_i \times \mathcal{P}_i \rightarrow \mathbb{R}^{n_i} \text{ given as } M(\bm{q}_i, \mathcal{P}_i) = \hat{\bm{r}}_i \equiv \left(\hat{r}_{i1}, \ldots, \hat{r}_{in_i}\right),
$$
where $\hat{\bm{r}}_i$ denotes the vector predicted relevance scores of the hotels in the $i$th query. The optimal model produces relevance scores such that the order of elements in $\hat{\bm{r}}_i$ is exactly equal to the order of the true relevance scores $\bm{r}_i$, after both are ranked in descending order based on their values.

Clearly, a model which produces the optimal ranking for every query, as described in the previous section, would be nearly impossible to build. Therefore, we need a way to evaluate a prediction model based on the ranking it produces; a metric that would be suitable is monotonically increasing or decreasing as in the similarity of the predicted ranking compared to the true ranking. A popular metric that is used to this end is the Normalized Discounted Cumulative Gain (NDCG). Let $\bm{r}_i^s$ and $\bm{\hat{r}}_i^s$ denote the vector of true relevance scores for the $i$th query, sorted by the true relevance and the predicted relevance, respectively. The Discounted Cumulative Gain (DCG) and Ideal Discounted Cumulative Gain (IDCG) for the $i$th query are then defined as:
$$
    DCG_i = \sum_{j=1}^{n_i} \frac{\bm{\hat{r}}_{ij}^s}{\log_2 (j + 1)}, \quad IDCG_i = \sum_{j=1}^{n_i} \frac{\bm{r}_{ij}^s}{\log_2 (j + 1)}.
$$
The NDCG for the $i$th query is then defined as follows:
$$
    NDCG_i = \frac{DCG_i}{IDCG_i}.
$$
Instead of constructing our evaluation metric on the entire list of items, we could also evaluate it only based on the $k$ most important items in the query, yielding the NDCG@$k$, which is obtained by replacing $n_i$ with $k$ in the equation above and recalculating the NDCG. For our purpose, we employ the NDCG@38, thus considering the top 38 items in the ranking. The NDCG@38 can be calculated for every query and their respective prediction from our model, the final evaluation score of the model on the test set is then calculated as the average NDCG@38 score over all queries $q_i \in \mathcal{Q}$ in the test set.

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



