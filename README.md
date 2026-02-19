# ECS171-Group8-models

1. Model list
ML: SVM, Random Forest, Naive Bayes, Logistic Regression
DL: CNNs, RNNs, LSTM

2. Model Development
can write a <.py> file under </Models>, or write it directly in a <.ipynb> under </ModelTraining>
3. Model Training
can copy exists file and edit by yourself (DL: LSTM; FE_ML: TFDIF_RF; Regular ML: RF)
4. Model Evalution
Accuracy, Precision, Recall, F1-Score, Confusion Matrix (TP/FP/FN/TN), ?approach graph

5. Google Colab
Use Google Colab to train DL
https://colab.research.google.com/drive/10JfBXe7yFZ9PQePn2sgqSAUkc1vDvBpj?usp=sharing
please train on a copy of this notebook, then update the result on branch by 
! git add . or ! git add <filename>
! git commit -m ""
! git push


// tree Feb18 5:27PM
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
