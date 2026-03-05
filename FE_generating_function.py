import re
import string
import pandas as pd
import nltk

# Download necessary NLTK data for VADER and POS tagging
# This might take a moment if not already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger.zip')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng.zip')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from scipy.sparse import hstack
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# take original text and timestamp
def FE_dataset_generating_function(text, timestamp):
    """
    Generate a feature-rich DataFrame from the original text and timestamp.
    Inputs: text (str), timestamp (datetime-like)
    Output: DataFrame with engineered features
        variables: ["original_text", 
        "hour", "month", "season", 
        "exclamation_count", "question_count", 
        "word_count", "char_count", "all_caps_words", 
        "uppercase_ratio", "total_punctuation", "avg_word_length", 
        "NN_count", "JJ_count", "VB_count", "RB_count",
        "DT_count", "PRP_count", "PUNCT_count", 
        "neg", "neu", "pos", "compound"]
    """
    # timestamp conversion
    timestamp = pd.to_datetime(timestamp)
    hour = timestamp.hour
    month = timestamp.month
    season = month.map({
        3: 1, 4: 1, 5: 1,
        6: 2, 7: 2, 8: 2,
        9: 3, 10: 3, 11: 3,
        12: 4, 1: 4, 2: 4
    })

    # feature engineer from text
    text = "" if pd.isna(text) else str(text)

    words = re.findall(r"\b\w+\b", text)
    caps_words = re.findall(r"\b[A-Z]{2,}\b", text)  # words fully uppercase (len>=2)

    char_count = len(text)
    letter_count = sum(ch.isalpha() for ch in text)
    uppercase_letters = sum(ch.isupper() for ch in text)

    #POS tags
    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    pos_counts = {
        'NN_count': 0, 'JJ_count': 0, 'VB_count': 0, 'RB_count': 0, # Nouns, Adjectives, Verbs, Adverbs
        'DT_count': 0, 'PRP_count': 0, # Determiners, Pronouns
        'PUNCT_count': 0 # Punctuation
    }
    for word, tag in tagged:
        if tag.startswith('NN'): pos_counts['NN_count'] += 1
        elif tag.startswith('JJ'): pos_counts['JJ_count'] += 1
        elif tag.startswith('VB'): pos_counts['VB_count'] += 1
        elif tag.startswith('RB'): pos_counts['RB_count'] += 1
        elif tag.startswith('DT'): pos_counts['DT_count'] += 1
        elif tag.startswith('PRP'): pos_counts['PRP_count'] += 1
        elif tag in ['.', ',', '!', '?']: pos_counts['PUNCT_count'] += 1
    pos_counts = pd.Series(pos_counts)

    # VADER sentiment scores
    # Initialize VADER sentiment intensity analyzer
    sia = SentimentIntensityAnalyzer()
    vs = sia.polarity_scores(text)

    # Concatenate POS features to the main dataframe
    df = pd.DataFrame([{
        "original_text": text, 
        "hour": hour, 
        "month": month, 
        "season": season,
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "word_count": len(words),
        "char_count": char_count,
        "all_caps_words": len(caps_words),
        "uppercase_ratio": (uppercase_letters / letter_count) if letter_count > 0 else 0.0,
        "total_punctuation": sum(ch in string.punctuation for ch in text),
        "avg_word_length": (sum(len(w) for w in words) / len(words)) if words else 0.0,
        **pos_counts,
        "neg": vs["neg"], "neu": vs["neu"], "pos": vs["pos"], "compound": vs["compound"]
    }])
    return df