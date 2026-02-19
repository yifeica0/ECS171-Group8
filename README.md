# ECS171-Group8-models

everytime before editing, `git pull` <br>
everytime after editing, 
```
git add .
git commit -m ""
git push
```
**1. Model list**<br>
ML: SVM, Random Forest, Naive Bayes, Logistic Regression<br>
DL: CNNs, RNNs, LSTM

**2. datasets:**<br>
Regualar ML: <datasets/amazon_user_reviews_features_sentiment.parquet><br>
FE+ML & DL: <datasets/amazon_user_reviews_text_sentiment.parquet>

**3. Model Development**<br>
can write a <.py> file under </Models>, or write it directly in a <.ipynb> under </ModelTraining>

**4. Model Training**<br>
can copy exists file and edit by yourself (DL: LSTM; FE_ML: TFDIF_RF; Regular ML: RF) under </ModelTraining>

**5. Model Evalution**<br>
Accuracy, Precision, Recall, F1-Score, Confusion Matrix (TP/FP/FN/TN), ?approach graph

**6. DL Training**<br>
Use Google Colab to train DL<br>
https://colab.research.google.com/drive/10JfBXe7yFZ9PQePn2sgqSAUkc1vDvBpj?usp=sharing<br>
please train on a copy of this notebook, then update the result on branch by 
```
! git add . or ! git add <filename>
! git commit -m ""
! git push
```
\
\
// tree Feb18 5:27PM
```
.
├── datasets
│   ├── amazon_user_reviews_3cat.parquet
│   ├── amazon_user_reviews_features_sentiment.parquet
│   └── amazon_user_reviews_text_sentiment.parquet
├── EDA
│   └── FeatureSelection_AllFeature_correlation_3cat.ipynb
├── Models
│   ├── DL
│   │   ├── __pycache__
│   │   │   └── LSTM.cpython-313.pyc
│   │   └── LSTM.py
│   └── ML
│       ├── __pycache__
│       │   └── RandomForest.cpython-313.pyc
│       └── RandomForest.py
├── ModelTraining
│   ├── DL
│   │   └── LSTM.ipynb
│   ├── FE_ML
│   │   └── TFDIF_RF.ipynb
│   └── Regular_ML
│       └── RF.ipynb
├── README.md
└── Result_ModelEvaluation
    └── results.csv
```
