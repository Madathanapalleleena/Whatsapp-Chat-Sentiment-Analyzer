"""
Topic Extraction using Latent Dirichlet Allocation (LDA).
Pure scikit-learn – no model downloads required.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


class TopicExtractor:
    def __init__(self, n_topics: int = 5):
        self.n_topics = n_topics
        self._vectorizer = CountVectorizer(
            max_df=0.90, min_df=2, max_features=500, stop_words='english'
        )
        self._lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=42, max_iter=8, n_jobs=-1
        )
        self.is_fitted = False

    # ---- fit ----

    def fit(self, texts: list[str]) -> None:
        valid = [t for t in texts if isinstance(t, str) and len(t) > 10]
        if len(valid) < max(self.n_topics * 3, 15):
            return
        try:
            X = self._vectorizer.fit_transform(valid)
            self._lda.fit(X)
            self.is_fitted = True
        except Exception as e:
            print(f"[TopicExtractor] fit failed: {e}")

    # ---- inference ----

    def get_topics(self, n_words: int = 10) -> dict[str, list[str]]:
        if not self.is_fitted:
            return {}
        feature_names = self._vectorizer.get_feature_names_out()
        return {
            f'Topic {i + 1}': [
                feature_names[j] for j in comp.argsort()[: -n_words - 1 : -1]
            ]
            for i, comp in enumerate(self._lda.components_)
        }

    def get_topic_labels(self, n_words: int = 3) -> dict[str, str]:
        return {k: ', '.join(v[:n_words]) for k, v in self.get_topics(n_words).items()}

    def predict_topic(self, text: str) -> int:
        if not self.is_fitted or not isinstance(text, str) or len(text) < 5:
            return 0
        try:
            X = self._vectorizer.transform([text])
            dist = self._lda.transform(X)[0]
            return int(np.argmax(dist)) + 1
        except Exception:
            return 0

    # ---- DataFrame ----

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        texts = df['processed_message'].fillna('').tolist()
        # Cap at 3000 messages for speed on large chats
        self.fit(texts[:3000])
        df['topic'] = df['processed_message'].apply(self.predict_topic)
        return df
