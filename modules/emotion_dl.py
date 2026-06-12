"""
Module 4 – Deep Learning Emotion Detection
  • Model : bhadresh-savani/distilbert-base-uncased-emotion (HuggingFace)
  • Labels: sadness | joy | love | anger | fear | surprise
  • Downloads ~250 MB on first run; subsequent runs use the local cache.
"""

from __future__ import annotations
import pandas as pd


_MODEL_ID = 'bhadresh-savani/distilbert-base-uncased-emotion'
_EMOTIONS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
_EMPTY = {e: 0.0 for e in _EMOTIONS}


class DLEmotionDetector:
    def __init__(self):
        self._pipe = None
        self._available = True

    @property
    def pipeline(self):
        if self._pipe is None and self._available:
            try:
                from transformers import pipeline
                self._pipe = pipeline(
                    'text-classification',
                    model=_MODEL_ID,
                    return_all_scores=True,
                    truncation=True,
                    max_length=512,
                )
            except Exception as e:
                self._available = False
                print(f"[EmotionDetector] Could not load model: {e}")
        return self._pipe

    def predict(self, text: str) -> dict[str, float]:
        """Return a dict of emotion → score for a single message."""
        if not text or len(str(text).strip()) < 3 or not self.pipeline:
            return dict(_EMPTY)
        try:
            results = self.pipeline(str(text)[:512])[0]
            return {r['label']: round(r['score'], 4) for r in results}
        except Exception:
            return dict(_EMPTY)

    def dominant_emotion(self, text: str) -> str:
        emotions = self.predict(text)
        return max(emotions, key=emotions.get) if any(emotions.values()) else 'neutral'

    def analyze_dataframe(
        self, df: pd.DataFrame, progress_callback=None
    ) -> pd.DataFrame:
        df = df.copy()
        emotions_list: list[dict] = []

        total = len(df)
        for i, text in enumerate(df['message']):
            emotions_list.append(self.predict(text))
            if progress_callback:
                progress_callback((i + 1) / total)

        df['emotions'] = emotions_list
        df['dominant_emotion'] = df['emotions'].apply(
            lambda e: max(e, key=e.get) if any(e.values()) else 'neutral'
        )
        return df
