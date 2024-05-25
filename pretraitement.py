from joblib import load
import re, nltk
nltk.download("stopwords")
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer



def cleaned_text(text: str):
    text = str(text)
    clean = re.sub(r"\n", " ", text)
    clean = clean.lower()
    clean = re.sub(r"[~.,%/:;?_&+*=!-]", " ", clean)
    clean = re.sub(r"[^a-z]", " ", clean)
    clean = clean.strip()
    clean = re.sub(r"\s{2,}", " ", clean)
    return clean
    