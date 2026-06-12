"""
Conversation Analytics – statistics, time patterns, word/emoji frequency.
"""

from __future__ import annotations
from collections import Counter
import pandas as pd
import numpy as np


class ConversationAnalytics:

    def basic_stats(self, df: pd.DataFrame) -> dict:
        span = (df['datetime'].max() - df['datetime'].min()).days
        stats = {
            'total_messages': len(df),
            'total_participants': df['sender'].nunique(),
            'date_range_days': max(span, 1),
            'avg_messages_per_day': round(len(df) / max(span, 1), 1),
            'messages_per_participant': df['sender'].value_counts().to_dict(),
        }
        if 'word_count' in df.columns:
            stats['avg_word_count'] = round(df['word_count'].mean(), 1)
            stats['total_words'] = int(df['word_count'].sum())
        return stats

    def activity_heatmap(self, df: pd.DataFrame) -> pd.DataFrame:
        tmp = df.copy()
        tmp['hour'] = tmp['datetime'].dt.hour
        tmp['day'] = tmp['datetime'].dt.day_name()
        pivot = tmp.pivot_table(
            index='day', columns='hour', values='message',
            aggfunc='count', fill_value=0
        )
        order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return pivot.reindex([d for d in order if d in pivot.index])

    def word_frequency(self, df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
        all_text = ' '.join(df['processed_message'].fillna('').tolist())
        freq = Counter(all_text.split()).most_common(top_n)
        return pd.DataFrame(freq, columns=['word', 'count'])

    def emoji_frequency(self, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        if 'emojis' not in df.columns:
            return pd.DataFrame(columns=['emoji', 'count'])
        all_emojis = ''.join(df['emojis'].fillna('').tolist())
        freq = Counter(all_emojis).most_common(top_n)
        return pd.DataFrame(freq, columns=['emoji', 'count'])

    def response_times(self, df: pd.DataFrame) -> dict[str, float]:
        tmp = df.sort_values('datetime').copy()
        tmp['resp_min'] = tmp['datetime'].diff().dt.total_seconds() / 60
        tmp = tmp[(tmp['resp_min'] > 0) & (tmp['resp_min'] < 120)]
        return tmp.groupby('sender')['resp_min'].mean().round(1).to_dict()

    def sentiment_over_time(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'sentiment_compound' not in df.columns:
            return pd.DataFrame()
        tmp = df.copy()
        tmp['date'] = tmp['datetime'].dt.date
        return tmp.groupby('date')['sentiment_compound'].mean().reset_index()

    def peak_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        tmp = df.copy()
        tmp['hour'] = tmp['datetime'].dt.hour
        return tmp.groupby('hour').size().reset_index(name='count')

    def daily_activity(self, df: pd.DataFrame) -> pd.DataFrame:
        tmp = df.copy()
        tmp['date'] = tmp['datetime'].dt.date
        return tmp.groupby('date').size().reset_index(name='count')
