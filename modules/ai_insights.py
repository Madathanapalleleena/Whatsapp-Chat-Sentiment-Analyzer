"""
AI Insights Generator using Google Gemini.
Falls back to a statistical summary when no API key is provided.
Supports real-time streaming via the Gemini streaming API.
"""

from __future__ import annotations
from typing import Generator
import pandas as pd


class AIInsightsGenerator:
    MODEL = 'gemini-2.0-flash'

    def __init__(self, api_key: str | None = None):
        self._model = None
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self._genai = genai
                self._model = genai.GenerativeModel(self.MODEL)
            except Exception as e:
                print(f'[AIInsightsGenerator] Could not init Gemini client: {e}')

    # ── helpers ────────────────────────────────────────────────────────────────

    def _build_prompt(self, df: pd.DataFrame, n_sample: int = 60) -> str:
        sample = df.tail(n_sample)
        chat_lines = [f"{r['sender']}: {r['message']}" for _, r in sample.iterrows()]

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

    def _build_data_context(self, df: pd.DataFrame) -> str:
        """Build a compact system context for the conversational AI assistant."""
        participants = df['sender'].unique().tolist()
        date_min = df['datetime'].min().date()
        date_max = df['datetime'].max().date()
        sent_dist = df['sentiment'].value_counts().to_dict()
        msg_counts = df['sender'].value_counts().to_dict()

        ctx = (
            "You are an AI assistant analyzing a WhatsApp conversation. "
            "Answer questions concisely, referencing the data when relevant.\n\n"
            "DATA SUMMARY:\n"
            f"- Total messages: {len(df):,}\n"
            f"- Participants: {', '.join(participants)}\n"
            f"- Date range: {date_min} to {date_max}\n"
            f"- Messages per person: {msg_counts}\n"
            f"- Sentiment distribution: {sent_dist}"
        )

        if 'dominant_emotion' in df.columns and df['dominant_emotion'].nunique() > 1:
            ctx += f"\n- Emotion distribution: {df['dominant_emotion'].value_counts().to_dict()}"

        if 'topic' in df.columns:
            ctx += f"\n- Topic distribution: {df['topic'].value_counts().to_dict()}"

        sample_rows = df.tail(40)[['sender', 'message']].values.tolist()
        ctx += "\n\nRECENT MESSAGES SAMPLE:\n"
        ctx += '\n'.join(f"{r[0]}: {r[1]}" for r in sample_rows)

        return ctx

    @staticmethod
    def _to_gemini_history(history: list[dict]) -> list[dict]:
        """Convert {"role": "user"|"assistant"} format to Gemini's {"role": "user"|"model"} format."""
        return [
            {'role': 'user' if m['role'] == 'user' else 'model', 'parts': [m['content']]}
            for m in history
        ]

    # ── public: standard (non-streaming) ───────────────────────────────────────

    def generate_summary(self, df: pd.DataFrame) -> str:
        if self._model:
            try:
                response = self._model.generate_content(self._build_prompt(df))
                return response.text
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

    def _build_relationship_prompt(self, df: pd.DataFrame) -> str:
        participants = df['sender'].unique().tolist()
        date_min = df['datetime'].min().date()
        date_max = df['datetime'].max().date()
        span_days = max((pd.Timestamp(date_max) - pd.Timestamp(date_min)).days, 1)

        msg_counts = df['sender'].value_counts().to_dict()

        avg_words = {}
        if 'word_count' in df.columns:
            for p in participants:
                avg_words[p] = round(df[df['sender'] == p]['word_count'].mean(), 1)

        sent_by_person = {}
        if 'sentiment' in df.columns:
            for p in participants:
                g = df[df['sender'] == p]
                sent_by_person[p] = g['sentiment'].mode()[0] if len(g) > 0 else 'N/A'

        peak_hours = {}
        for p in participants:
            g = df[df['sender'] == p]
            if len(g) > 0:
                peak_hours[p] = int(g['datetime'].dt.hour.mode()[0])

        tmp = df.sort_values('datetime').copy()
        tmp['resp_min'] = tmp['datetime'].diff().dt.total_seconds() / 60
        tmp = tmp[(tmp['resp_min'] > 0) & (tmp['resp_min'] < 120)]
        avg_response = tmp.groupby('sender')['resp_min'].mean().round(1).to_dict()

        sample = df.sample(min(80, len(df)), random_state=42).sort_values('datetime')
        chat_lines = [f"{r['sender']}: {r['message']}" for _, r in sample.iterrows()]

        return f"""Analyze the relationship between the participants in this WhatsApp conversation.

STATS:
- Participants: {', '.join(participants)}
- Chat span: {span_days} days ({date_min} → {date_max})
- Messages sent: {msg_counts}
- Avg words per message: {avg_words}
- Dominant sentiment per person: {sent_by_person}
- Peak activity hour: {peak_hours}
- Avg response time (min): {avg_response}

CONVERSATION SAMPLE:
{chr(10).join(chat_lines)}

Give a warm, human, and insightful relationship analysis. Cover these sections:

1. **Relationship Type** — What kind of relationship is this? (close friends, romantic, family, colleagues, acquaintances…). Give your reasoning from the chat signals.
2. **How Close Are They?** — Describe the bond and intimacy level.
3. **Communication Style** — How do they talk to each other? Playful, supportive, formal, teasing?
4. **Who Reaches Out More?** — Who initiates, who replies faster, who writes more?
5. **Emotional Dynamic** — What emotions run through the relationship?
6. **One Surprising Observation** — Something unique or unexpected about how they interact.

Be warm and conversational — like a friend explaining it, not a report."""

    def _fallback_relationship_analysis(self, df: pd.DataFrame) -> str:
        participants = df['sender'].unique().tolist()
        span = max((df['datetime'].max() - df['datetime'].min()).days, 1)
        counts = df['sender'].value_counts()
        total = len(df)

        lines = ['**Relationship Snapshot** *(add a Gemini API key for a full AI-powered analysis)*', '']

        if span > 180:
            lines.append(f'- **Long-running connection** — {span} days of chat history suggests a sustained relationship.')
        elif span > 30:
            lines.append(f'- **Established contact** — {span} days of conversation.')
        else:
            lines.append(f'- **Recent connection** — only {span} days of history so far.')

        if len(participants) == 2:
            a, b = counts.index[0], counts.index[1]
            pct = round(counts.iloc[0] / total * 100)
            if pct > 65:
                lines.append(f'- **{a}** drives the conversation ({pct}% of messages) — they tend to initiate more.')
            else:
                lines.append(f'- Well-balanced exchange — both participants contribute roughly equally.')

        tmp = df.sort_values('datetime').copy()
        tmp['resp_min'] = tmp['datetime'].diff().dt.total_seconds() / 60
        tmp = tmp[(tmp['resp_min'] > 0) & (tmp['resp_min'] < 30)]
        if not tmp.empty:
            avg_rt = tmp['resp_min'].mean()
            if avg_rt < 5:
                lines.append('- **Very fast replies** — they stay engaged and rarely leave each other waiting.')
            elif avg_rt < 15:
                lines.append('- **Reasonably quick replies** — a comfortable back-and-forth rhythm.')
            else:
                lines.append('- **Relaxed pace** — they take their time replying, suggesting a low-pressure dynamic.')

        night_msgs = df[df['datetime'].dt.hour.between(22, 23) | df['datetime'].dt.hour.between(0, 3)]
        if len(night_msgs) / total > 0.15:
            lines.append('- **Late-night chats** — a notable portion of messages happen after 10 PM, hinting at a close, personal bond.')

        if 'sentiment' in df.columns:
            pos_pct = round((df['sentiment'] == 'Positive').sum() / total * 100)
            if pos_pct > 60:
                lines.append(f'- **Positive atmosphere** — {pos_pct}% of messages are positive, reflecting a warm and supportive relationship.')
            elif pos_pct < 30:
                lines.append(f'- **Tense undertones** — relatively few positive messages; there may be friction in this relationship.')

        return '\n'.join(lines)

    # ── public: streaming ──────────────────────────────────────────────────────

    def stream_relationship_analysis(self, df: pd.DataFrame) -> Generator[str, None, None]:
        if not self._model:
            yield self._fallback_relationship_analysis(df)
            return
        try:
            response = self._model.generate_content(
                self._build_relationship_prompt(df),
                stream=True,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f'*Streaming error: {e}*\n\n'
            yield self._fallback_relationship_analysis(df)

    def stream_summary(self, df: pd.DataFrame) -> Generator[str, None, None]:
        """Stream the chat summary token-by-token using the Gemini streaming API."""
        if not self._model:
            yield self._fallback_summary(df)
            return

        try:
            response = self._model.generate_content(
                self._build_prompt(df),
                stream=True,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f'*Streaming error: {e}*\n\n'
            yield self._fallback_summary(df)

    def stream_answer(
        self,
        df: pd.DataFrame,
        question: str,
        history: list[dict],
    ) -> Generator[str, None, None]:
        """
        Stream an answer to a user question about the chat data.
        `history` is a list of {"role": "user"|"assistant", "content": str} dicts.
        """
        if not self._model:
            yield (
                "Please add your **Gemini API key** in the sidebar to use the AI Assistant. "
                "The key is never stored — it's only used for the current session."
            )
            return

        system_ctx = self._build_data_context(df)

        try:
            import google.generativeai as genai
            contextual_model = genai.GenerativeModel(
                self.MODEL,
                system_instruction=system_ctx,
            )
            gemini_history = self._to_gemini_history(history)
            chat = contextual_model.start_chat(history=gemini_history)
            response = chat.send_message(question, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f'*Error: {e}*'
