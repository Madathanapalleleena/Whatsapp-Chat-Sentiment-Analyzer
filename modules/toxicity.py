"""
Toxicity Analysis using Detoxify (Jigsaw multi-label classifier).
Downloads ~200 MB on first run.
"""

from __future__ import annotations
import pandas as pd

_EMPTY = {
    'toxicity': 0.0,
    'severe_toxicity': 0.0,
    'obscene': 0.0,
    'threat': 0.0,
    'insult': 0.0,
    'identity_attack': 0.0,
}

_THRESHOLD = 0.5


class ToxicityAnalyzer:
    def __init__(self):
        self._model = None
        self._available = True

    @property
    def model(self):
        if self._model is None and self._available:
            try:
                from detoxify import Detoxify
                self._model = Detoxify('original')
            except Exception as e:
                self._available = False
                print(f"[ToxicityAnalyzer] Could not load model: {e}")
        return self._model

    def analyze(self, text: str) -> dict[str, float]:
        if not text or len(str(text).strip()) < 3 or not self.model:
            return dict(_EMPTY)
        try:
            result = self.model.predict(str(text)[:512])
            return {k: round(float(v), 4) for k, v in result.items()}
        except Exception:
            return dict(_EMPTY)

    def analyze_dataframe(self, df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
        df = df.copy()
        scores_list: list[dict] = []

        total = len(df)
        for i, text in enumerate(df['message']):
            scores_list.append(self.analyze(text))
            if progress_callback:
                progress_callback((i + 1) / total)

        df['toxicity_scores'] = scores_list
        df['toxicity'] = df['toxicity_scores'].apply(lambda s: s.get('toxicity', 0.0))
        df['is_toxic'] = df['toxicity'] > _THRESHOLD
        df['toxicity_label'] = df['is_toxic'].map({True: 'Toxic', False: 'Clean'})
        return df
