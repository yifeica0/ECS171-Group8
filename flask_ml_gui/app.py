from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

app = Flask(__name__)

# =====================
# LOAD MODEL
# =====================
model = tf.keras.models.load_model("textcnn_model.keras")

with open("tokenizer.json") as f:
    tokenizer = tokenizer_from_json(f.read())

with open("label_map.json") as f:
    label_map = json.load(f)

reverse_map = {v: k for k, v in label_map.items()}

MAX_LEN = 250

# =====================
# ROUTES
# =====================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("input")

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    probs = model.predict(padded)
    pred_class = np.argmax(probs)

    sentiment = reverse_map[int(pred_class)]

    return render_template("result.html", prediction=sentiment)


if __name__ == "__main__":
    app.run(debug=True)