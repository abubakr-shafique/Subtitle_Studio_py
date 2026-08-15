"""Text-to-speech dubbing: turn translated subtitles into spoken audio.

Two engines mirror the translation backends:
  - EdgeTTSEngine: free Microsoft Edge neural voices (online, huge language coverage)
  - XTTSEngine:    Coqui XTTS v2 (offline, clones the original speaker's voice)

Dubbing places each synthesized cue at its original start time on a silent
canvas the length of the source media. A clip longer than its slot (up to the
next cue) is faded out and truncated; gaps stay silent.
"""
import asyncio
import tempfile
import wave
from pathlib import Path

from .media import normalize_wav
from .subtitles import parse_srt

SAMPLE_RATE = 24000

TTS_ENGINE_LABELS = {
    "edge": "Edge TTS (online)",
    "xtts": "XTTS v2 (offline, voice clone)",
}

# Default neural voice per target language (keyed by Google/ISO code).
# Run `edge-tts --list-voices` to find alternatives.
EDGE_VOICES = {
    "en": "en-US-JennyNeural",
    "tr": "tr-TR-EmelNeural",
    "es": "es-ES-ElviraNeural",
    "hi": "hi-IN-SwaraNeural",
    "ur": "ur-PK-UzmaNeural",
    "fr": "fr-FR-DeniseNeural",
    "zh-CN": "zh-CN-XiaoxiaoNeural",
    "zh-TW": "zh-TW-HsiaoChenNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "ar": "ar-SA-ZariyahNeural",
    "de": "de-DE-KatjaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "it": "it-IT-ElsaNeural",
    "nl": "nl-NL-ColetteNeural",
    "bn": "bn-IN-TanishaaNeural",
    "pa": "pa-IN-OjasNeural",
    "fa": "fa-IR-DilaraNeural",
    "vi": "vi-VN-HoaiMyNeural",
    "th": "th-TH-PremwadeeNeural",
    "id": "id-ID-GadisNeural",
    "pl": "pl-PL-ZofiaNeural",
    "uk": "uk-UA-PolinaNeural",
}

# Google/ISO code -> XTTS v2 language code (XTTS supports 17 languages; no Urdu).
XTTS_LANG_MAP = {
    "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it", "pt": "pt",
    "pl": "pl", "tr": "tr", "ru": "ru", "nl": "nl", "ar": "ar", "ja": "ja",
    "ko": "ko", "hi": "hi", "zh-cn": "zh-cn",
}


def make_tts_engine(engine_key, device="auto", log=print, reference_wav=None):
    if engine_key == "edge":
        return EdgeTTSEngine(log)
    if engine_key == "xtts":
        if not reference_wav:
            raise RuntimeError(
                "XTTS voice cloning needs the original media as a voice reference. "
                "For .srt input, use the Edge TTS engine instead."
            )
        return XTTSEngine(device, log, reference_wav)
    raise ValueError(f"Unknown TTS engine: {engine_key}")


class EdgeTTSEngine:
    """Microsoft Edge neural TTS via edge-tts. Free, requires internet."""

    def __init__(self, log=print):
        try:
            import edge_tts  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Edge TTS needs edge-tts. Run: pip install edge-tts") from exc
        self.log = log

    def check_language(self, lang_code):
        if lang_code not in EDGE_VOICES:
            raise ValueError(
                f"No default Edge voice for {lang_code!r}. Add one in core/tts.py (EDGE_VOICES)."
            )

    def synthesize(self, text, lang_code, out_stem, work_dir):
        import edge_tts
        dest = Path(work_dir) / (Path(out_stem).name + ".mp3")
        asyncio.run(edge_tts.Communicate(text, EDGE_VOICES[lang_code]).save(str(dest)))
        return dest


class XTTSEngine:
    """Coqui XTTS v2: offline, clones the reference speaker. CPML (non-commercial) license."""

    MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

    def __init__(self, device="auto", log=print, reference_wav=None):
        try:
            import torch
            from TTS.api import TTS
        except ImportError as exc:
            raise RuntimeError(
                "XTTS needs the Coqui TTS package. Run: pip install coqui-tts"
            ) from exc
        import os
        os.environ.setdefault("COQUI_TTS_AGREED", "1")  # accept the CPML model license
        use_cuda = device in ("auto", "cuda") and torch.cuda.is_available()
        log(f"Loading XTTS v2 on {'GPU' if use_cuda else 'CPU'} (downloads ~1.8 GB on first run)...")
        self.tts = TTS(self.MODEL_NAME)
        if use_cuda:
            self.tts.to("cuda")
        self.reference_wav = str(reference_wav)

    def check_language(self, lang_code):
        if lang_code.lower() not in XTTS_LANG_MAP:
            raise ValueError(
                f"XTTS v2 does not support {lang_code!r} "
                f"(supports: {', '.join(sorted(XTTS_LANG_MAP))}). Use Edge TTS instead."
            )

    def synthesize(self, text, lang_code, out_stem, work_dir):
        dest = Path(work_dir) / (Path(out_stem).name + ".wav")
        self.tts.tts_to_file(
            text=text,
            file_path=str(dest),
            speaker_wav=self.reference_wav,
            language=XTTS_LANG_MAP[lang_code.lower()],
        )
        return dest


# ---------------- audio helpers ----------------

def trim_wav(src_path, dst_path, seconds=30):
    """Keep only the first `seconds` of a PCM WAV (used as the XTTS voice reference)."""
    with wave.open(str(src_path), "rb") as w:
        params = w.getparams()
        frames = w.readframes(min(w.getnframes(), int(seconds * w.getframerate())))
    with wave.open(str(dst_path), "wb") as w:
        w.setparams(params)
        w.writeframes(frames)
    return Path(dst_path)


def _load_samples(path, work_dir):
    """Read a clip as int16 mono samples at SAMPLE_RATE, converting via ffmpeg if needed."""
    import numpy as np
    p = Path(path)
    try:
        with wave.open(str(p), "rb") as w:
            if (w.getsampwidth(), w.getnchannels(), w.getframerate()) == (2, 1, SAMPLE_RATE):
                return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    except wave.Error:
        pass
    fixed = Path(work_dir) / (p.stem + "_norm.wav")
    normalize_wav(p, fixed, SAMPLE_RATE)
    with wave.open(str(fixed), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def _write_wav(path, samples):
    import numpy as np
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(np.asarray(samples, dtype=np.int16).tobytes())


def dub_srt(srt_path, out_path, engine, lang_code, total_duration,
            log=print, cancel=None, progress_cb=None):
    """Synthesize every cue of a translated SRT and align it to the original timeline.

    Failsafe: a cue whose synthesis or mixing fails is left silent and the job
    continues with the remaining cues.
    """
    import numpy as np
    subs = parse_srt(Path(srt_path).read_text(encoding="utf-8-sig"))
    if not subs:
        raise ValueError(f"No subtitle cues could be parsed from {Path(srt_path).name}.")
    total = max(float(total_duration or 0.0), subs[-1].end) + 0.5
    canvas = np.zeros(int(total * SAMPLE_RATE), dtype=np.int16)
    spoken = failed = 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, cue in enumerate(subs):
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Cancelled during dubbing.")
            text = cue.text.strip()
            if text:
                try:
                    raw = engine.synthesize(text, lang_code, td / f"cue_{i:05d}", td)
                    samples = _load_samples(raw, td)
                    start_idx = min(int(cue.start * SAMPLE_RATE), len(canvas))
                    nxt = subs[i + 1].start if i + 1 < len(subs) else None
                    slot_end = (min(int(nxt * SAMPLE_RATE), len(canvas))
                                if nxt is not None else len(canvas))
                    avail = slot_end - start_idx
                    if avail > 0:
                        seg = samples[:avail]
                        if len(samples) > avail:  # too long for the slot: fade + truncate
                            seg = seg.copy()
                            fade = min(int(0.04 * SAMPLE_RATE), len(seg))
                            if fade > 0:
                                seg[-fade:] = (seg[-fade:].astype(np.float32)
                                               * np.linspace(1.0, 0.0, fade)).astype(np.int16)
                        mixed = (canvas[start_idx:start_idx + len(seg)].astype(np.int32)
                                 + seg.astype(np.int32))
                        canvas[start_idx:start_idx + len(seg)] = np.clip(
                            mixed, -32768, 32767).astype(np.int16)
                        spoken += 1
                    else:
                        log(f"WARNING: no room on the timeline for cue {i + 1}; skipped.")
                except Exception as exc:
                    failed += 1
                    log(f"WARNING: TTS failed for cue {i + 1} ({exc}); slot left silent.")
            if progress_cb:
                progress_cb((i + 1) / len(subs))
    out_path = Path(out_path)
    _write_wav(out_path, canvas)
    log(f"Dubbed audio written: {spoken} cue(s) spoken, {failed} skipped.")
    return out_path
