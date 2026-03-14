# ECS171-Group8-models

**1. Model list**<br>
ML: SVM, Random Forest, Naive Bayes, Logistic Regression<br>
DL: CNNs, RNNs, LSTM, BERT

**2. Datasets:**<br>
200 each sentiment category, 600 in total per product category (34 categories in tlotal)<br>
ML: quantative features + lineared review texts<br>
 <datasets/amazon_user_reviews_text_sentiment_with_sw.parquet><br>
DL: review texts<br>
<datasets/amazon_user_reviews_text_sentiment_with_sw.parquet><br>

**3. Feature Engineering:**<br>
for ML: <br>
linearization of review texts: TF-IDF<br>
engineerization of quantative feaures: Multicollinearity Deletion (95% percentile threshold) + Standardization + PCA<br>
![list of quantative features](/images/CorrelationBetweenFeatures.png)
for DL:<br>
raw tokenized review text

**4. Model Evalution**<br>
Accuracy, Precision, Recall, F1-Score, Confusion Matrix (TP/FP/FN/TN), ROC curve<br>

**5.Demo Running**<br>
To run the demo: <br>
Use python 3.11 version to run it. Before run it, make sure you have installed all the extra packages like TensorFlow.<br>
Go into the demo repository and use terminal to run the command below:<br>
```shell
py -3.11 app.py
```
Wait for 5-10 seconds, then it will run the website in local. Open the link in the terminal and select a model you want to use. Type the review text you want to predict and press the botton at the bottom, the model will show result of sentiment prediction.
