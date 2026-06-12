import re
import nltk
import emoji as emoji_lib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd


def _download_nltk():
    for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass


_download_nltk()

_URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
_PUNCT_RE = re.compile(r'[^\w\s]')
_DIGITS_RE = re.compile(r'\d+')
_SPACES_RE = re.compile(r'\s+')


class TextPreprocessor:
    def __init__(self):
        try:
            self._stop_words = set(stopwords.words('english'))
        except Exception:
            self._stop_words = set()
        self._lemmatizer = WordNetLemmatizer()

    # --- public helpers ---

    def extract_emojis(self, text: str) -> str:
        return ''.join(ch for ch in text if ch in emoji_lib.EMOJI_DATA)

    def remove_urls(self, text: str) -> str:
        return _URL_RE.sub('', text)

    def remove_emojis(self, text: str) -> str:
        return emoji_lib.replace_emoji(text, replace='')

    def clean_text(self, text: str) -> str:
        text = self.remove_urls(text)
        text = self.remove_emojis(text)
        text = _PUNCT_RE.sub(' ', text)
        text = _DIGITS_RE.sub('', text)
        text = text.lower()
        return _SPACES_RE.sub(' ', text).strip()

    def tokenize_and_lemmatize(self, text: str) -> str:
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
        tokens = [
            self._lemmatizer.lemmatize(t)
            for t in tokens
            if t not in self._stop_words and len(t) > 2
        ]
        return ' '.join(tokens)

    def preprocess(self, text: str) -> str:
        return self.tokenize_and_lemmatize(self.clean_text(text))

    # --- DataFrame-level ---

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['emojis'] = df['message'].apply(self.extract_emojis)
        df['cleaned_message'] = df['message'].apply(self.clean_text)
        df['processed_message'] = df['cleaned_message'].apply(self.tokenize_and_lemmatize)
        df['word_count'] = df['message'].apply(lambda x: len(str(x).split()))
        df['char_count'] = df['message'].apply(len)
        return df
