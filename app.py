"""
WhatsApp Chat Analyzer – Streamlit Dashboard
Run: streamlit run app.py
"""

import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from modules.parser import parse_whatsapp_chat
from modules.preprocessor import TextPreprocessor
from modules.sentiment_ml import MLSentimentAnalyzer
from modules.emotion_dl import DLEmotionDetector
from modules.toxicity import ToxicityAnalyzer
from modules.topics import TopicExtractor
from modules.ai_insights import AIInsightsGenerator
from modules.analytics import ConversationAnalytics

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title='WhatsApp Chat Analyzer',
    page_icon=None,
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown(
    """
    <style>
    /* ── Typography ── */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* ── Layout ── */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* ── Page title ── */
    h1 {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
        letter-spacing: -0.5px;
        margin-bottom: 0.25rem !important;
    }

    /* ── Section headers ── */
    h2, h3 {
        font-weight: 600 !important;
        color: #1f2937 !important;
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem 1.25rem;
    }
    [data-testid="metric-container"] label {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7280 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid #e5e7eb;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem;
        font-weight: 500;
        color: #6b7280;
        padding: 0.6rem 1.1rem;
        border: none;
        border-bottom: 2px solid transparent;
        margin-bottom: -2px;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #111827 !important;
        border-bottom: 2px solid #111827 !important;
        font-weight: 600 !important;
        background: transparent !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #f9fafb;
        border-right: 1px solid #e5e7eb;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280 !important;
        font-weight: 600 !important;
    }

    /* ── Info / success / warning boxes ── */
    [data-testid="stAlert"] {
        border-radius: 6px;
        font-size: 0.85rem;
    }

    /* ── Divider ── */
    hr { border-color: #e5e7eb !important; }

    /* ── Buttons ── */
    .stButton > button {
        font-weight: 500;
        border-radius: 6px;
        font-size: 0.85rem;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        overflow: hidden;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploadDropzone"] {
        border-radius: 8px;
        border: 1.5px dashed #d1d5db;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SENTIMENT_COLORS = {'Positive': '#16a34a', 'Neutral': '#6b7280', 'Negative': '#dc2626'}
CHART_COLORS = ['#2563eb', '#7c3aed', '#0891b2', '#d97706', '#059669', '#dc2626']
PLOTLY_LAYOUT = dict(
    paper_bgcolor='white',
    plot_bgcolor='white',
    font=dict(family='Inter, Segoe UI, sans-serif', size=12, color='#374151'),
    title_font=dict(size=14, color='#111827', family='Inter, Segoe UI, sans-serif'),
    margin=dict(t=40, b=30, l=10, r=10),
    xaxis=dict(showgrid=True, gridcolor='#f3f4f6', zeroline=False),
    yaxis=dict(showgrid=True, gridcolor='#f3f4f6', zeroline=False),
)

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner='Loading emotion model...')
def _emotion_detector():
    try:
        d = DLEmotionDetector()
        _ = d.pipeline
        return d
    except Exception:
        return None


@st.cache_resource(show_spinner='Loading toxicity model...')
def _toxicity_analyzer():
    try:
        a = ToxicityAnalyzer()
        _ = a.model
        return a
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner='Parsing and preprocessing...')
def _parse_and_preprocess(content: str) -> pd.DataFrame:
    df = parse_whatsapp_chat(content)
    if df.empty:
        return df
    return TextPreprocessor().preprocess_dataframe(df)


@st.cache_data(show_spinner='Running sentiment analysis...')
def _run_sentiment(content: str) -> pd.DataFrame:
    df = _parse_and_preprocess(content)
    if df.empty:
        return df
    return MLSentimentAnalyzer().analyze_dataframe(df)


@st.cache_data(show_spinner='Extracting topics...')
def _run_topics(content: str) -> tuple[pd.DataFrame, dict, dict]:
    df = _run_sentiment(content)
    if df.empty:
        return df, {}, {}
    extractor = TopicExtractor(n_topics=5)
    df = extractor.analyze_dataframe(df)
    return df, extractor.get_topics(), extractor.get_topic_labels()


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _apply_layout(fig, **overrides):
    layout = {**PLOTLY_LAYOUT, **overrides}
    fig.update_layout(**layout)
    return fig


def _wordcloud(text: str):
    wc = WordCloud(
        width=1000, height=380,
        background_color='white',
        colormap='Blues',
        max_words=120,
        prefer_horizontal=0.85,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    fig.patch.set_facecolor('white')
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _load_api_key() -> str | None:
    """Read Gemini API key from Streamlit secrets or environment — never shown to users."""
    try:
        return st.secrets['GEMINI_API_KEY']
    except Exception:
        return os.getenv('GEMINI_API_KEY') or None


def _sidebar() -> tuple[bool, bool]:
    with st.sidebar:
        st.markdown('### Analysis Settings')

        enable_emotion = st.toggle(
            'Emotion Detection (BERT)',
            value=False,
            help='DistilBERT model — downloads ~250 MB on first use. Disable on low-memory environments.',
        )
        enable_toxicity = st.toggle(
            'Toxicity Analysis',
            value=False,
            help='Detoxify model — downloads ~200 MB on first use',
        )

        st.divider()

        st.divider()

        st.markdown('### Export Instructions')
        st.markdown(
            '1. Open a WhatsApp conversation\n'
            '2. Tap the menu and select **Export chat**\n'
            '3. Choose **Without Media**\n'
            '4. Save and upload the `.txt` file above'
        )

        st.divider()

        if st.button('Load Sample Chat', use_container_width=True):
            sample = os.path.join(os.path.dirname(__file__), 'data', 'sample_chat.txt')
            if os.path.exists(sample):
                with open(sample, encoding='utf-8') as f:
                    st.session_state['sample_content'] = f.read()
                st.success('Sample chat loaded.')

    return enable_emotion, enable_toxicity


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

def _tab_overview(df: pd.DataFrame, stats: dict, analytics: ConversationAnalytics):
    st.subheader('Conversation Overview')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Messages', f"{stats['total_messages']:,}")
    c2.metric('Participants', stats['total_participants'])
    c3.metric('Days Active', stats['date_range_days'])
    c4.metric('Avg Words / Message', stats.get('avg_word_count', '—'))

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        vc = df['sender'].value_counts()
        fig = px.pie(
            values=vc.values, names=vc.index,
            title='Message Share by Participant',
            hole=0.45,
            color_discrete_sequence=CHART_COLORS,
        )
        _apply_layout(fig, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        daily = analytics.daily_activity(df)
        fig = px.line(
            daily, x='date', y='count',
            title='Daily Message Volume',
            labels={'count': 'Messages', 'date': 'Date'},
            color_discrete_sequence=[CHART_COLORS[0]],
        )
        fig.update_traces(line_width=2)
        _apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    ph = analytics.peak_hours(df)
    fig = px.bar(
        ph, x='hour', y='count',
        title='Activity by Hour of Day',
        labels={'hour': 'Hour (24 h)', 'count': 'Messages'},
        color='count',
        color_continuous_scale='Blues',
    )
    _apply_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

    all_text = ' '.join(df['processed_message'].dropna())
    if all_text.strip():
        st.subheader('Word Cloud')
        _wordcloud(all_text)

    wf = analytics.word_frequency(df, top_n=20)
    if not wf.empty:
        fig = px.bar(
            wf, x='count', y='word', orientation='h',
            title='Top 20 Words',
            color='count',
            color_continuous_scale='Blues',
            labels={'count': 'Frequency', 'word': ''},
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        _apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    ef = analytics.emoji_frequency(df, top_n=10)
    if not ef.empty:
        st.subheader('Top Emojis Used')
        cols = st.columns(len(ef))
        for i, row in ef.iterrows():
            with cols[i]:
                st.markdown(
                    f"<div style='font-size:26px;text-align:center;padding:4px'>{row['emoji']}</div>"
                    f"<div style='text-align:center;font-size:11px;color:#6b7280'>{row['count']}</div>",
                    unsafe_allow_html=True,
                )


def _tab_sentiment(df: pd.DataFrame, analytics: ConversationAnalytics):
    st.subheader('Sentiment Analysis')
    st.caption(
        'Pipeline: TF-IDF (unigrams + bigrams) — Logistic Regression, '
        'bootstrapped with VADER labels from the chat corpus.'
    )

    col1, col2 = st.columns(2)

    with col1:
        sc = df['sentiment'].value_counts()
        fig = px.bar(
            x=sc.index, y=sc.values,
            title='Sentiment Distribution',
            color=sc.index,
            color_discrete_map=SENTIMENT_COLORS,
            labels={'x': 'Sentiment', 'y': 'Messages'},
        )
        _apply_layout(fig, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if df['sender'].nunique() > 1:
            grp = df.groupby(['sender', 'sentiment']).size().unstack(fill_value=0)
            fig = px.bar(
                grp, title='Sentiment by Participant',
                barmode='group',
                color_discrete_map=SENTIMENT_COLORS,
            )
            _apply_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

    trend = analytics.sentiment_over_time(df)
    if not trend.empty:
        fig = px.line(
            trend, x='date', y='sentiment_compound',
            title='Sentiment Trend Over Time',
            labels={'sentiment_compound': 'Compound Score', 'date': 'Date'},
            color_discrete_sequence=[CHART_COLORS[0]],
        )
        fig.update_traces(line_width=2)
        fig.add_hline(y=0, line_dash='dash', line_color='#9ca3af', opacity=0.7)
        fig.add_hrect(y0=0.05, y1=1.0, fillcolor='#16a34a', opacity=0.05, line_width=0)
        fig.add_hrect(y0=-1.0, y1=-0.05, fillcolor='#dc2626', opacity=0.05, line_width=0)
        _apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Message Highlights')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('**Most Positive Messages**')
        top = df.nlargest(5, 'sentiment_compound')[['sender', 'message', 'sentiment_compound']]
        st.dataframe(top.reset_index(drop=True), use_container_width=True)
    with col2:
        st.markdown('**Most Negative Messages**')
        bot = df.nsmallest(5, 'sentiment_compound')[['sender', 'message', 'sentiment_compound']]
        st.dataframe(bot.reset_index(drop=True), use_container_width=True)


def _tab_emotions(df: pd.DataFrame):
    st.subheader('Emotion Detection')

    if 'dominant_emotion' not in df.columns or df['dominant_emotion'].nunique() <= 1:
        st.info('Enable Emotion Detection in the sidebar, then re-upload the file.')
        return

    st.caption(
        'Model: bhadresh-savani/distilbert-base-uncased-emotion '
        '(DistilBERT fine-tuned on 6 classes: joy, sadness, anger, fear, love, surprise)'
    )

    col1, col2 = st.columns(2)

    with col1:
        ec = df['dominant_emotion'].value_counts()
        fig = px.pie(
            values=ec.values, names=ec.index,
            hole=0.4,
            title='Emotion Distribution',
            color_discrete_sequence=CHART_COLORS,
        )
        _apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if df['sender'].nunique() > 1:
            grp = df.groupby(['sender', 'dominant_emotion']).size().unstack(fill_value=0)
            fig = px.bar(
                grp, title='Emotions by Participant',
                barmode='stack',
                color_discrete_sequence=CHART_COLORS,
            )
            _apply_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

    tmp = df.copy()
    tmp['date'] = tmp['datetime'].dt.date
    et = tmp.groupby(['date', 'dominant_emotion']).size().unstack(fill_value=0).reset_index()
    if not et.empty:
        emotion_cols = [c for c in et.columns if c != 'date']
        fig = px.area(
            et, x='date', y=emotion_cols,
            title='Emotion Trends Over Time',
            color_discrete_sequence=CHART_COLORS,
        )
        _apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)


def _tab_toxicity(df: pd.DataFrame):
    st.subheader('Toxicity Analysis')

    if 'toxicity' not in df.columns:
        st.info('Enable Toxicity Analysis in the sidebar, then re-upload the file.')
        return

    st.caption('Model: Detoxify — Jigsaw multi-label toxicity classifier')

    toxic = df['is_toxic'].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric('Toxic Messages', int(toxic))
    c2.metric('Clean Messages', len(df) - int(toxic))
    c3.metric('Toxicity Rate', f'{toxic / len(df) * 100:.1f}%')

    fig = px.histogram(
        df, x='toxicity', nbins=50,
        title='Toxicity Score Distribution',
        color_discrete_sequence=['#dc2626'],
        labels={'toxicity': 'Toxicity Score'},
    )
    _apply_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

    if df['sender'].nunique() > 1:
        tbu = df.groupby('sender')['toxicity'].mean().reset_index()
        fig = px.bar(
            tbu, x='sender', y='toxicity',
            title='Average Toxicity Score by Participant',
            color='toxicity',
            color_continuous_scale='RdYlGn_r',
        )
        _apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    flagged = df[df['is_toxic']][['sender', 'message', 'toxicity']].head(15)
    if not flagged.empty:
        st.markdown('**Flagged Messages**')
        st.dataframe(flagged.reset_index(drop=True), use_container_width=True)
    else:
        st.success('No highly toxic messages detected.')


def _tab_topics(df: pd.DataFrame, topics: dict, topic_labels: dict):
    st.subheader('Topic Extraction')
    st.caption('Model: Latent Dirichlet Allocation (LDA) with CountVectorizer features')

    if not topics:
        st.warning('Insufficient vocabulary for topic modeling (minimum 15 messages required).')
        return

    cols = st.columns(min(5, len(topics)))
    for i, (name, words) in enumerate(topics.items()):
        with cols[i % len(cols)]:
            label = topic_labels.get(name, name)
            st.markdown(
                f"<div style='background:#f9fafb;border:1px solid #e5e7eb;"
                f"border-radius:8px;padding:14px 16px;height:200px'>"
                f"<div style='font-size:0.75rem;font-weight:700;text-transform:uppercase;"
                f"letter-spacing:0.06em;color:#6b7280;margin-bottom:6px'>{name}</div>"
                f"<div style='font-size:0.8rem;color:#374151;font-style:italic;"
                f"margin-bottom:8px'>{label}</div>"
                f"<div style='font-size:0.82rem;color:#1f2937'>"
                + ''.join(f"<div style='padding:2px 0;border-bottom:1px solid #f3f4f6'>{w}</div>"
                          for w in words[:6])
                + "</div></div>",
                unsafe_allow_html=True,
            )

    st.divider()

    if 'topic' in df.columns:
        tc = df['topic'].value_counts().reset_index()
        tc.columns = ['topic_num', 'count']
        tc['label'] = tc['topic_num'].apply(
            lambda n: topic_labels.get(f'Topic {n}', f'Topic {n}')
        )
        fig = px.bar(
            tc, x='label', y='count',
            title='Messages per Topic',
            color='count',
            color_continuous_scale='Blues',
            labels={'label': 'Topic', 'count': 'Messages'},
        )
        _apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)


def _tab_ai_insights(df: pd.DataFrame, analytics: ConversationAnalytics, api_key: str | None):
    st.subheader('AI-Generated Insights')
    gen = AIInsightsGenerator(api_key=api_key)

    st.markdown('#### Participant Profiles')
    profiles = gen.relationship_insights(df)
    cols = st.columns(min(3, len(profiles)))
    for i, (name, data) in enumerate(profiles.items()):
        with cols[i % len(cols)]:
            st.markdown(
                f"<div style='background:#ffffff;border:1px solid #e5e7eb;"
                f"border-radius:8px;padding:16px 20px'>"
                f"<div style='font-weight:700;font-size:1rem;color:#111827;"
                f"margin-bottom:12px;border-bottom:1px solid #f3f4f6;padding-bottom:8px'>{name}</div>"
                f"<div style='font-size:0.82rem;color:#374151;line-height:1.8'>"
                f"<b>Messages:</b> {data['message_count']}<br>"
                f"<b>Avg length:</b> {data['avg_words']} words<br>"
                f"<b>Sentiment:</b> {data['dominant_sentiment']}<br>"
                + (f"<b>Emotion:</b> {data['dominant_emotion'].capitalize()}<br>"
                   if data['dominant_emotion'] not in ('N/A', 'neutral') else '')
                + f"<b>Peak hour:</b> {data['peak_hour']}:00"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown('#### Response Times')
    rt = analytics.response_times(df)
    if rt:
        rt_df = pd.DataFrame(rt.items(), columns=['Participant', 'Avg Response (min)'])
        st.dataframe(rt_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown('#### Chat Summary')
    if st.button('Generate Summary', type='primary'):
        if api_key:
            st.write_stream(gen.stream_summary(df))
        else:
            with st.spinner('Generating summary...'):
                st.markdown(gen.generate_summary(df))
    else:
        note = (
            'Click **Generate Summary** to get a Gemini-powered analysis with real-time streaming.'
            if api_key else
            'Click **Generate Summary** for a statistical summary, '
            'or add your Gemini API key in the sidebar for a live AI-powered version.'
        )
        st.info(note)


def _tab_ai_assistant(df: pd.DataFrame, api_key: str | None, content_hash: int):
    st.subheader('AI Assistant')
    st.caption(
        'Conversational AI powered by Claude — ask anything about your chat. '
        'Supports voice input and real-time streaming responses.'
    )

    gen = AIInsightsGenerator(api_key=api_key)

    # Per-file chat history
    hist_key = f'ai_chat_{content_hash}'
    if hist_key not in st.session_state:
        st.session_state[hist_key] = []
    messages: list[dict] = st.session_state[hist_key]

    # Display existing conversation
    for msg in messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # ── Voice input ────────────────────────────────────────────────────────────
    with st.expander('Voice Input', expanded=False):
        st.caption('Record your question and click **Transcribe & Ask**.')
        audio_input = st.audio_input('Record a question')

        if audio_input is not None:
            col1, col2 = st.columns([1, 3])
            if col1.button('Transcribe & Ask', type='primary'):
                try:
                    from modules.voice import transcribe_audio
                    with st.spinner('Transcribing...'):
                        question = transcribe_audio(audio_input.read())
                    st.session_state[f'{hist_key}_voice_q'] = question
                    st.success(f'Transcribed: "{question}"')
                    st.rerun()
                except Exception as e:
                    st.error(f'Transcription failed: {e}')

    # Pick up a pending voice question from the previous rerun
    voice_q_key = f'{hist_key}_voice_q'
    pending_voice = st.session_state.pop(voice_q_key, None)

    # ── Text input ─────────────────────────────────────────────────────────────
    text_prompt = st.chat_input('Ask something about your chat…')

    question = pending_voice or text_prompt
    if not question:
        if not messages:
            st.markdown(
                "<div style='color:#6b7280;font-size:0.85rem;padding:8px 0'>"
                "Try asking: <i>\"Who messages the most?\", \"What's the overall mood?\", "
                "\"Summarize the last week of messages\"</i>"
                "</div>",
                unsafe_allow_html=True,
            )
        return

    # Display user turn
    with st.chat_message('user'):
        st.markdown(question)
    messages.append({'role': 'user', 'content': question})

    # Build history for context (last 8 turns, excluding the one just added)
    history_for_api = [
        {'role': m['role'], 'content': m['content']}
        for m in messages[:-1][-8:]
    ]

    # Stream assistant response
    with st.chat_message('assistant'):
        response: str = st.write_stream(gen.stream_answer(df, question, history_for_api))

    messages.append({'role': 'assistant', 'content': response})

    # Voice output
    try:
        from modules.voice import text_to_speech_bytes
        audio_bytes = text_to_speech_bytes(response)
        st.audio(audio_bytes, format='audio/mp3')
    except Exception:
        pass  # Voice output is optional; silently skip if gTTS unavailable

    # Clear history button
    if messages and st.button('Clear conversation', key=f'clear_{hist_key}'):
        st.session_state[hist_key] = []
        st.rerun()


def _tab_raw(df: pd.DataFrame):
    st.subheader('Analyzed Data')

    col1, col2 = st.columns(2)
    with col1:
        users = st.multiselect(
            'Filter by participant',
            df['sender'].unique().tolist(),
            default=df['sender'].unique().tolist(),
        )
    with col2:
        sentiments = st.multiselect(
            'Filter by sentiment',
            ['Positive', 'Neutral', 'Negative'],
            default=['Positive', 'Neutral', 'Negative'],
        )

    filtered = df[df['sender'].isin(users) & df['sentiment'].isin(sentiments)]

    display_cols = ['datetime', 'sender', 'message', 'sentiment', 'sentiment_compound']
    for opt in ('dominant_emotion', 'toxicity', 'is_toxic', 'topic'):
        if opt in filtered.columns:
            display_cols.append(opt)

    st.dataframe(filtered[display_cols].reset_index(drop=True), use_container_width=True)
    st.download_button(
        'Download as CSV',
        data=filtered[display_cols].to_csv(index=False),
        file_name='whatsapp_analysis.csv',
        mime='text/csv',
    )


# ---------------------------------------------------------------------------
# Landing page (no file uploaded)
# ---------------------------------------------------------------------------

def _landing():
    st.markdown('---')
    c1, c2, c3 = st.columns(3)

    card_style = (
        "background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;"
        "padding:24px 28px;height:200px"
    )
    heading_style = (
        "font-size:0.75rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.07em;color:#6b7280;margin-bottom:10px"
    )
    body_style = "font-size:0.85rem;color:#374151;line-height:1.9"

    with c1:
        st.markdown(
            f"<div style='{card_style}'>"
            f"<div style='{heading_style}'>Analytics</div>"
            f"<div style='{body_style}'>"
            "Activity heatmaps<br>Word clouds and frequency<br>"
            "Response time analysis<br>Emoji usage patterns"
            "</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div style='{card_style}'>"
            f"<div style='{heading_style}'>ML / DL Models</div>"
            f"<div style='{body_style}'>"
            "TF-IDF + Logistic Regression<br>DistilBERT emotion detection<br>"
            "LDA topic modeling<br>Detoxify toxicity scoring"
            "</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div style='{card_style}'>"
            f"<div style='{heading_style}'>AI Assistant</div>"
            f"<div style='{body_style}'>"
            "Voice input &amp; TTS output<br>Real-time streaming responses<br>"
            "Multi-turn conversation<br>Claude-powered summaries"
            "</div></div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title('WhatsApp Chat Analyzer')
    st.markdown(
        '<p style="color:#6b7280;font-size:0.95rem;margin-top:-8px">'
        'Upload a WhatsApp chat export to analyze sentiment, emotions, topics, and conversation patterns.'
        '</p>',
        unsafe_allow_html=True,
    )

    enable_emotion, enable_toxicity = _sidebar()
    api_key = _load_api_key()

    uploaded = st.file_uploader(
        'Chat Export File',
        type=['txt'],
        label_visibility='collapsed',
    )

    content: str | None = None
    if uploaded:
        content = uploaded.read().decode('utf-8', errors='ignore')
        st.session_state.pop('sample_content', None)
    elif 'sample_content' in st.session_state:
        content = st.session_state['sample_content']
        st.info('Using sample chat data. Upload your own file to analyze a real conversation.')

    if not content:
        _landing()
        return

    # Core pipeline
    df, topics, topic_labels = _run_topics(content)

    if df is None or df.empty:
        st.error('Could not parse the file. Please verify it is a valid WhatsApp chat export.')
        return

    # Optional heavy models
    state_key = f'analyzed_{hash(content)}'
    if state_key not in st.session_state:
        work_df = df.copy()

        if enable_emotion:
            detector = _emotion_detector()
            if detector:
                prog = st.progress(0, text='Running emotion detection...')
                work_df = detector.analyze_dataframe(
                    work_df,
                    progress_callback=lambda p: prog.progress(
                        p, text=f'Emotion detection — {p * 100:.0f}%'
                    ),
                )
                prog.empty()
            else:
                st.warning('Emotion Detection unavailable — torch/transformers not installed.')

        if enable_toxicity:
            analyzer = _toxicity_analyzer()
            if analyzer:
                prog = st.progress(0, text='Running toxicity analysis...')
                work_df = analyzer.analyze_dataframe(
                    work_df,
                    progress_callback=lambda p: prog.progress(
                        p, text=f'Toxicity analysis — {p * 100:.0f}%'
                    ),
                )
                prog.empty()
            else:
                st.warning('Toxicity Analysis unavailable — torch/detoxify not installed.')

        st.session_state[state_key] = work_df

    df = st.session_state[state_key]

    analytics = ConversationAnalytics()
    stats = analytics.basic_stats(df)

    st.markdown(
        f"<div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:6px;"
        f"padding:10px 16px;font-size:0.875rem;color:#166534;margin-bottom:1rem'>"
        f"Analysis complete &mdash; "
        f"<b>{stats['total_messages']:,} messages</b> from "
        f"<b>{stats['total_participants']} participants</b> "
        f"over <b>{stats['date_range_days']} days</b>."
        f"</div>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        'Overview', 'Sentiment', 'Emotions',
        'Toxicity', 'Topics', 'AI Insights', 'AI Assistant', 'Raw Data',
    ])

    content_hash = hash(content)

    with tabs[0]:
        _tab_overview(df, stats, analytics)
    with tabs[1]:
        _tab_sentiment(df, analytics)
    with tabs[2]:
        _tab_emotions(df)
    with tabs[3]:
        _tab_toxicity(df)
    with tabs[4]:
        _tab_topics(df, topics, topic_labels)
    with tabs[5]:
        _tab_ai_insights(df, analytics, api_key)
    with tabs[6]:
        _tab_ai_assistant(df, api_key, content_hash)
    with tabs[7]:
        _tab_raw(df)


if __name__ == '__main__':
    main()
