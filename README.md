# ECS171-Group8-models

everytime before editing, `git pull` <br>
everytime after editing, 
```
git add .
git commit -m "xx"
git push
```
**1. Model list**<br>
ML: SVM, Random Forest, Naive Bayes, Logistic Regression<br>
DL: CNNs, RNNs, LSTM

**2. datasets:**<br>
200 each sentiment category, 600 in total per product category<br>
sentiment category: rating1-2: neg (sentiment 2), rating3: neutral (sentiment 1), rating4-5: pos (sentiment 0)
\
Regualar ML: <datasets/amazon_user_reviews_features_sentiment.parquet><br>
FE+ML & DL: <datasets/amazon_user_reviews_text_sentiment.parquet><br>
\
extended datasets:<br>
400 each sentiment category, 1200 in total per product category<br>
Regualar ML: <datasets/extended_datasets/amazon_user_reviews_features_sentiment_extended.parquet><br>
FE+ML & DL: <datasets/extended_datasets/amazon_user_reviews_text_sentiment_extended.parquet><br>

**3. Model Development**<br>
can write a <.py> file under </Models>, or write it directly in a <.ipynb> under </ModelTraining>

**4. Model Training**<br>
can copy exists file and edit by yourself (DL: LSTM; FE_ML: TFDIF_RF; Regular ML: RF) under </ModelTraining>

**5. Model Evalution**<br>
Accuracy, Precision, Recall, F1-Score, Confusion Matrix (TP/FP/FN/TN), ROC curve
```
y_score = model.predict_proba(X_test)

import sys
import os
sys.path.insert(0, os.path.abspath('../../'))

# ModelEvaluation
from ModelEvaluation.ModelEvaluation import ModelEvaluation
# Model Evaluation
evaluator = ModelEvaluation()
evaluator.run_pipeline(y_test, y_pred, y_score)
```

**6. DL Training**<br>
Use Google Colab to train DL<br>
please upload your DL file on Google Colab, download the result, move it back to the folder, then update the result on branch by git<br>
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
