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
DL: CNNs, RNNs, LSTM<br>

---

**to retrain with dataset with stop word**<br>
please copy your model.ipynb into /with_sw folder and run<br>
e.g. `ModelTraining/FE_ML/TFIDF_NB.ipynb` --> `ModelTraining/FE_ML/with_sw/TFIDF_NB copy.ipynb`
<br>

---
**2. datasets:**<br>
200 each sentiment category, 600 in total per product category<br>
extended datasets: 400 each sentiment category, 1200 in total per product category<br>
**sentiment category: rating1-2: neg (sentiment 2), rating3: neutral (sentiment 1), rating4-5: pos (sentiment 0)**

----

textual dataset **with stop_word**:<br>
`../../../datasets/with_stop_word/amazon_user_reviews_text_sentiment_with_sw.parquet`<br>
with stop_word extended:<br>
`../../../datasets/with_stop_word/extended/amazon_user_reviews_text_sentiment_with_sw_extended.parquet`<br>

----

feature + textual dataset **with stop_word**<br>
`../../../datasets/with_stop_word/amazon_user_reviews_textANDfeature_sentiment_with_sw.parquet`
`../../../datasets/with_stop_word/extended/amazon_user_reviews_textANDfeature_sentiment_with_sw_extended.parquet`

----

Regualar ML: <datasets/amazon_user_reviews_features_sentiment.parquet><br>
Regualar ML: <datasets/extended_datasets/amazon_user_reviews_features_sentiment_extended.parquet><br>

----

textual dataset without stop word:
FE+ML & DL: <datasets/amazon_user_reviews_text_sentiment.parquet><br>
FE+ML & DL: <datasets/extended_datasets/amazon_user_reviews_text_sentiment_extended.parquet><br>

---
**3. Model Development**<br>
can write a <.py> file under </Models>, or write it directly in a <.ipynb> under </ModelTraining>

---
**4. Model Training**<br>
can copy exists file and edit by yourself (DL: LSTM; FE_ML: TFDIF_RF; Regular ML: RF) under </ModelTraining>

---
**5. Model Evalution**<br>
Accuracy, Precision, Recall, F1-Score, Confusion Matrix (TP/FP/FN/TN), ROC curve
```
y_score = model.predict_proba(X_test)

import sys
import os
sys.path.insert(0, os.path.abspath('../../'))
# if you are under /with_sw 
# sys.path.insert(0, os.path.abspath('../../../'))

# ModelEvaluation
from ModelEvaluation.ModelEvaluation import ModelEvaluation
evaluator = ModelEvaluation()
evaluator.run_pipeline(y_test, y_pred, y_score)
```
---
**6. DL Training**<br>
Use Google Colab to train DL<br>
please upload your DL file on Google Colab, download the result, move it back to the folder, then update the result on branch by git<br>

---
// tree Feb18 5:27PM
```
.
├── datasets
│   ├── amazon_user_reviews_3cat.parquet
│   ├── amazon_user_reviews_features_sentiment.parquet
│   ├── amazon_user_reviews_text_sentiment.parquet
│   ├── extended_datasets
│   │   ├── amazon_user_reviews_3cat_extended.parquet
│   │   ├── amazon_user_reviews_features_sentiment_extended.parquet
│   │   └── amazon_user_reviews_text_sentiment_extended.parquet
│   └── with_stop_word
│       ├── amazon_user_reviews_text_sentiment_with_sw.parquet
│       ├── amazon_user_reviews_textANDfeature_sentiment_with_sw.parquet
│       └── extended
│           ├── amazon_user_reviews_text_sentiment_with_sw_extended.parquet
│           └── amazon_user_reviews_textANDfeature_sentiment_with_sw_extended.parquet
├── EDA
│   ├── FeatureSelection_AllFeature_correlation_3cat_extended.ipynb
│   └── FeatureSelection_AllFeature_correlation_3cat.ipynb
├── ModelEvaluation
│   ├── __pycache__
│   │   └── ModelEvaluation.cpython-313.pyc
│   └── ModelEvaluation.py
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
│   │   ├── CNN.ipynb
│   │   ├── LSTM.ipynb
│   │   ├── RNN_grid_search.ipynb
│   │   ├── RNN.ipynb
│   │   └── with_sw
│   ├── FE_ML
│   │   ├── text_Bert.ipynb
│   │   ├── text_logistic_NB_RF.ipynb
│   │   ├── TFIDF_NB.ipynb
│   │   ├── TFIDF_RF.ipynb
│   │   ├── TFIDF_SVM.ipynb
│   │   ├── TFIDF_XGB.ipynb
│   │   └── with_sw
│   │       └── TFIDF_NB copy.ipynb
│   └── Regular_ML
│       ├── Chuyuan_copy.ipynb
│       ├── Chuyuan.ipynb
│       ├── NaiveBayes.ipynb
│       ├── RF.ipynb
│       ├── SVM.ipynb
│       ├── with_sw
│       └── XGBoost.ipynb
├── README.md
└── Result_ModelEvaluation
    └── results.csv

21 directories, 37 files
```
