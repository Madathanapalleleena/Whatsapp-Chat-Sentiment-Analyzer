"""
Module 3 – ML-Based Sentiment Analysis
  • Features  : TF-IDF (unigram + bigram)
  • Model     : Logistic Regression  (default) or Random Forest
  • Bootstrap : VADER labels are used as ground-truth when no external
                dataset is available, turning this into a semi-supervised
                self-training approach on the actual chat corpus.
"""

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


_LABEL_MAP = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}


class MLSentimentAnalyzer:
    def __init__(self, model_type: str = 'logistic_regression'):
        self._vader = SentimentIntensityAnalyzer()
        self._vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), sublinear_tf=True
        )
        if model_type == 'random_forest':
            self._model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            self._model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

        self.model_type = model_type
        self.is_trained = False

    # ---- private ----

    def _vader_label(self, text: str) -> int:
        score = self._vader.polarity_scores(str(text))['compound']
        if score >= 0.05:
            return 1
        if score <= -0.05:
            return -1
        return 0

    # ---- public ----

    def vader_scores(self, text: str) -> dict:
        return self._vader.polarity_scores(str(text))

    def train(self, messages: pd.Series) -> None:
        """Train on the chat corpus using VADER as the label source."""
        texts = messages.astype(str).tolist()
        if len(texts) < 20:
            return
        labels = [self._vader_label(t) for t in texts]
        # Only train if we have more than one class
        if len(set(labels)) < 2:
            return
        X = self._vectorizer.fit_transform(texts)
        self._model.fit(X, labels)
        self.is_trained = True

    def predict(self, texts) -> list[str]:
        if isinstance(texts, str):
            texts = [texts]
        texts = [str(t) for t in texts]

        if self.is_trained:
            X = self._vectorizer.transform(texts)
            return [_LABEL_MAP[p] for p in self._model.predict(X)]
        # Fallback: pure VADER
        return [_LABEL_MAP[self._vader_label(t)] for t in texts]

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        non_empty = df['message'][df['message'].astype(str).str.len() > 3]
        if len(non_empty) >= 20:
            self.train(non_empty)

        df['sentiment'] = self.predict(df['message'].tolist())
        df['sentiment_scores'] = df['message'].apply(self.vader_scores)
        df['sentiment_compound'] = df['sentiment_scores'].apply(lambda s: s['compound'])
        df['sentiment_positive'] = df['sentiment_scores'].apply(lambda s: s['pos'])
        df['sentiment_negative'] = df['sentiment_scores'].apply(lambda s: s['neg'])
        df['sentiment_neutral'] = df['sentiment_scores'].apply(lambda s: s['neu'])
        return df
