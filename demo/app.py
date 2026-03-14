from pathlib import Path
import json
import re
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from FE_generating_function import FE_dataset_generating_function, analyze_text
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
MAX_LEN = 250

LR_EXTRA_FEATURE_COLUMNS = [
    "hour", "month", "season",
    "exclamation_count", "question_count",
    "word_count", "char_count", "all_caps_words",
    "uppercase_ratio", "total_punctuation", "avg_word_length",
    "NN_count", "JJ_count", "VB_count", "RB_count",
    "DT_count", "PRP_count", "PUNCT_count",
    "neg", "neu", "pos", "compound"
]

MODEL_CONFIG = {
    'cnn': {
        'display_name': 'CNN Model',
        'type': 'cnn',
        'folder': MODELS_DIR / 'cnn',
    },
    'lr': {
        'display_name': 'Logistic Regression',
        'type': 'lr',
        'folder': MODELS_DIR / 'lr',
    },
}

# 0 = Positive, 1 = Neutral, 2 = Negative
SENTIMENT_NAMES = {
    0: 'Positive',
    1: 'Neutral',
    2: 'Negative',
}

loaded_models = {}
load_errors = {}


def _load_cnn_model(folder: Path):
    model_path = folder / 'textcnn_model.keras'
    tokenizer_path = folder / 'tokenizer.json'
    label_map_path = folder / 'label_map.json'

    model = tf.keras.models.load_model(model_path)

    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer = tokenizer_from_json(f.read())

    with open(label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)

    # label_map often looks like {"0": 0, "1": 1, "2": 2}
    reverse_map = {int(v): int(k) if str(k).isdigit() else k for k, v in label_map.items()}

    return {
        'kind': 'cnn',
        'model': model,
        'tokenizer': tokenizer,
        'reverse_map': reverse_map,
    }


def _load_lr_model(folder: Path):
    model_path = folder / 'logistic_model.joblib'
    vectorizer_path = folder / 'tfidf_vectorizer.joblib'

    lr_model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(vectorizer_path)

    return {
        'kind': 'lr',
        'model': lr_model,
        'vectorizer': tfidf_vectorizer,
    }


def load_all_models():
    loaded_models.clear()
    load_errors.clear()

    for model_key, config in MODEL_CONFIG.items():
        try:
            if config['type'] == 'cnn':
                loaded_models[model_key] = _load_cnn_model(config['folder'])
            else:
                loaded_models[model_key] = _load_lr_model(config['folder'])
        except Exception as e:
            load_errors[model_key] = f'{type(e).__name__}: {e}'


load_all_models()


def normalize_label(raw_label):
    """
    Convert model output label into integer 0/1/2 if possible.
    """
    if isinstance(raw_label, (int, np.integer)):
        return int(raw_label)

    raw_str = str(raw_label).strip()
    if raw_str.isdigit():
        return int(raw_str)

    lowered = raw_str.lower()
    if lowered == 'positive':
        return 0
    if lowered == 'neutral':
        return 1
    if lowered == 'negative':
        return 2

    return raw_label


def tokenize_for_display(text: str):
    """
    Split text into words and punctuation so the visualization looks better.
    Example: 'good!' -> ['good', '!']
    """
    return re.findall(r"\w+|[^\w\s]", text)


def score_to_strength(score: float, max_score: float):
    """
    Convert contribution magnitude to 0..1 for frontend heat intensity.
    """
    if max_score <= 1e-12:
        return 0.0
    return float(min(abs(score) / max_score, 1.0))


def predict_with_cnn(model_bundle, text: str):
    seq = model_bundle['tokenizer'].texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

    probs = model_bundle['model'].predict(padded, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))

    raw_label = model_bundle['reverse_map'].get(pred_idx, pred_idx)
    label = normalize_label(raw_label)
    sentiment = SENTIMENT_NAMES.get(label, str(label))

    return sentiment, confidence, pred_idx


def predict_with_lr(model_bundle, text: str):

    # TF-IDF features
    X_text = model_bundle['vectorizer'].transform([text])

    # Create timestamp (current time)
    timestamp = datetime.now()

    # Generate engineered features
    fe_df = FE_dataset_generating_function(text, timestamp)

    X_extra_dense = fe_df[LR_EXTRA_FEATURE_COLUMNS].astype(float).to_numpy()
    X_extra = csr_matrix(X_extra_dense)

    # Combine TF-IDF + engineered features
    X_final = hstack([X_text, X_extra])

    pred_class = int(model_bundle['model'].predict(X_final)[0])
    probs = model_bundle['model'].predict_proba(X_final)[0]
    confidence = float(np.max(probs))

    label = normalize_label(pred_class)
    sentiment = SENTIMENT_NAMES.get(label, str(label))

    return sentiment, confidence, pred_class


def get_cnn_probs(model_bundle, text: str):
    seq = model_bundle['tokenizer'].texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    probs = model_bundle['model'].predict(padded, verbose=0)[0]
    return probs


def explain_with_cnn(model_bundle, text):
    tokens = tokenize_for_display(text)

    original_probs = get_cnn_probs(model_bundle, text)
    pred_class = int(np.argmax(original_probs))
    original_score = original_probs[pred_class]

    predicted_label = model_bundle['reverse_map'].get(pred_class, pred_class)
    predicted_label = normalize_label(predicted_label)

    words = []

    for i, token in enumerate(tokens):
        reduced_text = " ".join(tokens[:i] + tokens[i+1:])
        new_probs = get_cnn_probs(model_bundle, reduced_text)
        new_score = new_probs[pred_class]

        contribution = original_score - new_score

        if contribution > 0.01:
            if predicted_label == 0:      # Positive
                sentiment = "good"
            elif predicted_label == 2:    # Negative
                sentiment = "bad"
            else:                         # Neutral
                sentiment = "neutral"
        elif contribution < -0.01:
            if predicted_label == 0:
                sentiment = "bad"
            elif predicted_label == 2:
                sentiment = "good"
            else:
                sentiment = "neutral"
        else:
            sentiment = "neutral"

        words.append({
            "word": token,
            "sentiment": sentiment,
            "score": float(abs(contribution))
        })

    return words

def explain_with_lr(model_bundle, text, pred_class):
    vectorizer = model_bundle["vectorizer"]
    model = model_bundle["model"]

    tokens = tokenize_for_display(text)

    X = vectorizer.transform([text])
    x_row = X.toarray()[0]

    vocab = vectorizer.vocabulary_
    coefs = model.coef_[pred_class]

    predicted_label = normalize_label(pred_class)

    words = []

    for token in tokens:
        key = token.lower()

        if key in vocab:
            idx = vocab[key]
            contribution = x_row[idx] * coefs[idx]
        else:
            contribution = 0

        if contribution > 0.01:
            if predicted_label == 0:      # Positive
                sentiment = "good"
            elif predicted_label == 2:    # Negative
                sentiment = "bad"
            else:
                sentiment = "neutral"
        elif contribution < -0.01:
            if predicted_label == 0:
                sentiment = "bad"
            elif predicted_label == 2:
                sentiment = "good"
            else:
                sentiment = "neutral"
        else:
            sentiment = "neutral"

        words.append({
            "word": token,
            "sentiment": sentiment,
            "score": float(abs(contribution))
        })

    return words

@app.route('/', methods=['GET'])
def index():
    return render_template(
        'index.html',
        model_config=MODEL_CONFIG,
        load_errors=load_errors,
        selected_model='cnn'
    )


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('input', '').strip()
    selected_model = request.form.get('model_choice', 'cnn')

    if not text:
        return render_template(
            'index.html',
            model_config=MODEL_CONFIG,
            load_errors=load_errors,
            selected_model=selected_model,
            error='Please enter some text before predicting.'
        )

    if selected_model not in MODEL_CONFIG:
        return render_template(
            'index.html',
            model_config=MODEL_CONFIG,
            load_errors=load_errors,
            selected_model='cnn',
            error='Invalid model selection.'
        )

    if selected_model in load_errors:
        return render_template(
            'index.html',
            model_config=MODEL_CONFIG,
            load_errors=load_errors,
            selected_model=selected_model,
            error=f"That model could not be loaded: {load_errors[selected_model]}"
        )

    model_bundle = loaded_models[selected_model]
    model_name = MODEL_CONFIG[selected_model]['display_name']

    try:
        if model_bundle['kind'] == 'cnn':
            prediction, confidence, pred_class = predict_with_cnn(model_bundle, text)
            words = explain_with_cnn(model_bundle, text)
        else:
            prediction, confidence, pred_class = predict_with_lr(model_bundle, text)
            words = explain_with_lr(model_bundle, text, pred_class)

        vader_words = analyze_text(text)        

        return render_template(
            'result.html',
            text=text,
            model_name=model_name,
            prediction=prediction,
            confidence=f'{confidence * 100:.1f}%',
            words=words,
            vader_words=vader_words,
            model_key=selected_model
        )

    except Exception as e:
        return render_template(
            'index.html',
            model_config=MODEL_CONFIG,
            load_errors=load_errors,
            selected_model=selected_model,
            error=f'Prediction failed: {type(e).__name__}: {e}'
        )


if __name__ == '__main__':
    app.run(debug=True)