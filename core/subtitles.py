"""Subtitle data model plus SRT/VTT reading and writing (no third-party deps)."""
import re
from collections import namedtuple
from pathlib import Path

Subtitle = namedtuple("Subtitle", ["index", "start", "end", "text"])

_TS_RE = re.compile(
    r"(?P<sh>\d{1,2}):(?P<sm>\d{2}):(?P<ss>\d{2})[,.](?P<sms>\d{3})\s*-->\s*"
    r"(?P<eh>\d{1,2}):(?P<em>\d{2}):(?P<es>\d{2})[,.](?P<ems>\d{3})"
)


def format_timestamp(seconds, sep=","):
    """Seconds -> 'HH:MM:SS,mmm' (SRT) or 'HH:MM:SS.mmm' (VTT)."""
    ms = int(round(float(seconds) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _to_seconds(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def compose_srt(subs):
    blocks = []
    for i, s in enumerate(subs, 1):
        blocks.append(
            f"{i}\n{format_timestamp(s.start, ',')} --> {format_timestamp(s.end, ',')}\n{s.text}"
        )
    return "\n\n".join(blocks) + "\n"


def parse_srt(text):
    """Parse SRT text into a list of Subtitle cues (tolerant of CRLF, BOM, missing indices)."""
    subs = []
    for block in re.split(r"\r?\n\s*\r?\n", text.strip()):
        lines = block.strip().splitlines()
        ts_idx = next((i for i, line in enumerate(lines) if _TS_RE.search(line)), None)
        if ts_idx is None:
            continue
        m = _TS_RE.search(lines[ts_idx])
        start = _to_seconds(m["sh"], m["sm"], m["ss"], m["sms"])
        end = _to_seconds(m["eh"], m["em"], m["es"], m["ems"])
        try:
            index = int(lines[0].strip()) if ts_idx > 0 else len(subs) + 1
        except ValueError:
            index = len(subs) + 1
        content = "\n".join(lines[ts_idx + 1:]).strip()
        subs.append(Subtitle(index, start, end, content))
    return subs


def write_srt(subs, path):
    path = Path(path)
    path.write_text(compose_srt(subs), encoding="utf-8")
    return path


def write_vtt(subs, path):
    lines = ["WEBVTT", ""]
    for s in subs:
        lines.append(f"{format_timestamp(s.start, '.')} --> {format_timestamp(s.end, '.')}")
        lines.append(s.text)
        lines.append("")
    path = Path(path)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
