import re
import pandas as pd
from datetime import datetime


# Patterns for Android (12h / 24h) and iOS WhatsApp exports
_PATTERNS = [
    # Android 12h:  12/01/25, 9:00 AM - Name: msg
    re.compile(
        r'^(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}),\s'
        r'(\d{1,2}:\d{2}\s?[AP]M)\s-\s([^:]+):\s(.*)$'
    ),
    # Android 24h:  12/01/25, 21:00 - Name: msg
    re.compile(
        r'^(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}),\s'
        r'(\d{1,2}:\d{2})\s-\s([^:]+):\s(.*)$'
    ),
    # iOS:  [12/01/25, 9:00:45 AM] Name: msg
    re.compile(
        r'^\[(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}),\s'
        r'(\d{1,2}:\d{2}(?::\d{2})?\s?[AP]M)\]\s([^:]+):\s(.*)$'
    ),
]

_SYSTEM_KEYWORDS = [
    'messages and calls are end-to-end encrypted',
    '<media omitted>',
    'image omitted',
    'video omitted',
    'audio omitted',
    'document omitted',
    'sticker omitted',
    'gif omitted',
    'contact card omitted',
    'this message was deleted',
    'you deleted this message',
    'missed voice call',
    'missed video call',
    'your security code with',
    'changed the subject to',
    'changed this group',
]

_DATETIME_FORMATS = [
    '%d/%m/%y %I:%M %p',
    '%d/%m/%Y %I:%M %p',
    '%m/%d/%y %I:%M %p',
    '%m/%d/%Y %I:%M %p',
    '%d/%m/%y %H:%M',
    '%d/%m/%Y %H:%M',
    '%m/%d/%y %H:%M',
    '%m/%d/%Y %H:%M',
    '%d.%m.%y %I:%M %p',
    '%d.%m.%Y %I:%M %p',
    '%d.%m.%y %H:%M',
    '%d.%m.%Y %H:%M',
    '%d/%m/%y %I:%M:%S %p',
    '%m/%d/%y %I:%M:%S %p',
]


def _try_parse_datetime(date_str: str, time_str: str) -> datetime | None:
    combined = f"{date_str} {time_str}".strip()
    for fmt in _DATETIME_FORMATS:
        try:
            return datetime.strptime(combined, fmt)
        except ValueError:
            continue
    return None


def _match_line(line: str):
    for pattern in _PATTERNS:
        m = pattern.match(line)
        if m:
            return m.groups()
    return None


def parse_whatsapp_chat(file_content: str) -> pd.DataFrame:
    """Parse a WhatsApp exported .txt file into a structured DataFrame."""
    messages = []
    current: dict | None = None

    for raw_line in file_content.splitlines():
        line = raw_line.strip()
        groups = _match_line(line)

        if groups:
            if current:
                messages.append(current)
            date_str, time_str, sender, message = groups
            current = {
                'date': date_str,
                'time': time_str.strip(),
                'sender': sender.strip(),
                'message': message.strip(),
            }
        elif current and line:
            current['message'] += ' ' + line

    if current:
        messages.append(current)

    if not messages:
        return pd.DataFrame()

    df = pd.DataFrame(messages)

    # Parse datetime
    df['datetime'] = df.apply(
        lambda r: _try_parse_datetime(r['date'], r['time']), axis=1
    )

    # Drop system / media messages
    mask = ~df['message'].str.lower().str.contains(
        '|'.join(_SYSTEM_KEYWORDS), na=False
    )
    df = df[mask].dropna(subset=['datetime']).reset_index(drop=True)

    return df
