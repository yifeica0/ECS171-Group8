Combined Flask app for 3 sentiment models.

Folder layout expected:

combined_app/
  app.py
  templates/
    index.html
    result.html
  models/
    1cnn/
      textcnn_model.keras
      tokenizer.json
      label_map.json
    2cnn/
      textcnn_model.keras
      tokenizer.json
      label_map.json
    3lr/
      logistic_model.joblib
      tfidf_vectorizer.joblib

Important:
The ZIP files you uploaded did not include the two CNN .keras model files, so this combined app will only fully run after you copy those files into:
  models/1cnn/textcnn_model.keras
  models/2cnn/textcnn_model.keras

How to run:
1. Install dependencies:
   pip install flask tensorflow joblib numpy scikit-learn
2. Put all model files into the folders above.
3. Run:
   python app.py
4. Open the local Flask address shown in the terminal.
