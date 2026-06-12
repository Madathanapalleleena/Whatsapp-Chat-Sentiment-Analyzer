"""
Voice utilities: speech-to-text (SpeechRecognition) and text-to-speech (gTTS).
Requires: SpeechRecognition, gTTS, pydub (+ ffmpeg system package for non-WAV input).
"""

from __future__ import annotations
import io


def transcribe_audio(audio_bytes: bytes) -> str:
    """Convert audio bytes to text using Google Speech Recognition."""
    import speech_recognition as sr

    recognizer = sr.Recognizer()

    # Try direct WAV read first; fall back to pydub for WebM/OGG from browsers
    try:
        audio_io = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_io) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except Exception:
        pass

    try:
        from pydub import AudioSegment
        audio_io = io.BytesIO(audio_bytes)
        segment = AudioSegment.from_file(audio_io)
        wav_io = io.BytesIO()
        segment.export(wav_io, format='wav')
        wav_io.seek(0)
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except Exception as e:
        raise RuntimeError(
            f"Could not transcribe audio: {e}. "
            "Try speaking more clearly or check your microphone."
        )


def text_to_speech_bytes(text: str, lang: str = 'en') -> bytes:
    """Convert text to MP3 audio bytes using Google TTS."""
    from gtts import gTTS

    # Trim to a reasonable length to avoid very long TTS output
    trimmed = text[:800]
    buf = io.BytesIO()
    gTTS(text=trimmed, lang=lang, slow=False).write_to_fp(buf)
    buf.seek(0)
    return buf.read()
