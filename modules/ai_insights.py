"""
AI Insights Generator using Claude (Anthropic).
Falls back to a statistical summary when no API key is provided.
"""

from __future__ import annotations
import pandas as pd


class AIInsightsGenerator:
    MODEL = 'claude-sonnet-4-6'

    def __init__(self, api_key: str | None = None):
        self._client = None
        if api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
            except Exception as e:
                print(f'[AIInsightsGenerator] Could not init Anthropic client: {e}')

    # ---- helpers ----

    def _build_prompt(self, df: pd.DataFrame, n_sample: int = 60) -> str:
        sample = df.tail(n_sample)
        chat_lines = [
            f"{r['sender']}: {r['message']}"
            for _, r in sample.iterrows()
        ]

        participants = df['sender'].unique().tolist()
        date_min = df['datetime'].min().date()
        date_max = df['datetime'].max().date()
        sent_dist = df['sentiment'].value_counts().to_dict()

        top_emotions = ''
        if 'dominant_emotion' in df.columns and df['dominant_emotion'].nunique() > 1:
            top_emotions = df['dominant_emotion'].value_counts().head(3).to_dict().__str__()

        return f"""Analyze this WhatsApp conversation and provide structured insights.

STATS:
- Messages: {len(df)} | Participants: {', '.join(participants)}
- Period: {date_min} → {date_max}
- Sentiment breakdown: {sent_dist}
- Top emotions: {top_emotions or 'N/A'}

RECENT SAMPLE:
{chr(10).join(chat_lines)}

Provide a concise response with these sections:
1. **Summary** – 2-3 sentences on what the conversation is about
2. **Key Themes** – bullet list of main topics discussed
3. **Relationship Dynamics** – how participants interact with each other
4. **Emotional Patterns** – notable emotional trends
5. **Interesting Observations** – anything unusual or noteworthy"""

    # ---- public ----

    def generate_summary(self, df: pd.DataFrame) -> str:
        if self._client:
            try:
                response = self._client.messages.create(
                    model=self.MODEL,
                    max_tokens=1024,
                    messages=[{'role': 'user', 'content': self._build_prompt(df)}],
                )
                return response.content[0].text
            except Exception as e:
                return f'*AI summary error: {e}*\n\n{self._fallback_summary(df)}'
        return self._fallback_summary(df)

    def _fallback_summary(self, df: pd.DataFrame) -> str:
        counts = df['sender'].value_counts()
        dominant_sentiment = (
            df['sentiment'].value_counts().index[0]
            if 'sentiment' in df.columns and len(df) > 0 else 'N/A'
        )
        lines = [
            '**Statistical Summary** *(no API key provided – add one for AI insights)*',
            '',
            f"- **Total messages**: {len(df):,}",
            f"- **Participants**: {', '.join(df['sender'].unique()[:6])}",
            f"- **Most active**: {counts.index[0]} ({counts.iloc[0]} messages)",
            f"- **Overall sentiment**: {dominant_sentiment}",
            f"- **Date range**: {df['datetime'].min().date()} → {df['datetime'].max().date()}",
        ]
        if 'dominant_emotion' in df.columns and df['dominant_emotion'].nunique() > 1:
            top_e = df['dominant_emotion'].value_counts().index[0]
            lines.append(f'- **Dominant emotion**: {top_e.capitalize()}')
        return '\n'.join(lines)

    def relationship_insights(self, df: pd.DataFrame) -> dict:
        result = {}
        for sender, group in df.groupby('sender'):
            result[sender] = {
                'message_count': len(group),
                'avg_words': round(group['word_count'].mean(), 1) if 'word_count' in group else 0,
                'dominant_sentiment': (
                    group['sentiment'].mode()[0]
                    if 'sentiment' in group.columns and len(group) > 0 else 'N/A'
                ),
                'dominant_emotion': (
                    group['dominant_emotion'].mode()[0]
                    if 'dominant_emotion' in group.columns and len(group) > 0 else 'N/A'
                ),
                'peak_hour': (
                    int(group['datetime'].dt.hour.mode()[0])
                    if len(group) > 0 else 0
                ),
            }
        return result
