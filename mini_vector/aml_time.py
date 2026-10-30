"""Observation time formatting shared by AML experiments; no clock fallback."""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from .prompts import _source_date

WEEKDAYS = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
MONTHS = ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
          'September', 'October', 'November', 'December')


def _sources(metadata: dict[str, Any] | None) -> list[Any]:
    metadata = metadata or {}
    stamps = metadata.get('source_timestamps')
    if isinstance(stamps, list):
        valid = [stamp for stamp in stamps if _source_date(stamp) is not None]
        if valid:
            return valid
    session = metadata.get('session_date')
    return [session] if _source_date(session) is not None else []


def observed_dates(metadata: dict[str, Any] | None) -> list[date]:
    """Prefer explicit source timestamps; use session_date when none are valid."""
    return sorted({date.fromisoformat(_source_date(value)) for value in _sources(metadata)})


def _english(value: date) -> str:
    return f'{WEEKDAYS[value.weekday()]}, {value.day} {MONTHS[value.month - 1]} {value.year}'


def date_prefix(metadata: dict[str, Any] | None) -> str:
    dates = observed_dates(metadata)
    if not dates:
        return ''
    if len(dates) == 1:
        return f'[Stated on {_english(dates[0])}] '
    return f'[Stated between {_english(dates[0])} and {_english(dates[-1])}] '


def observed_at(metadata: dict[str, Any] | None) -> str | None:
    values = []
    for value in _sources(metadata):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parsed = datetime.fromtimestamp(value / 1000, timezone.utc)
        else:
            try:
                parsed = datetime.strptime(value.strip(), '%I:%M %p on %d %B, %Y')
            except ValueError:
                parsed = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
        values.append(parsed)
    # Naive LoCoMo values preserve their representation; UTC is used only as a sort key.
    earliest = min(values, key=lambda dt: dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt) if values else None
    return earliest.isoformat() if earliest is not None else None
