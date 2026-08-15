"""Speech-to-text with faster-whisper (CTranslate2 backend)."""
from .subtitles import Subtitle


def _resolve_device(device):
    if device in ("cuda", "cpu"):
        return device
    try:
        import ctranslate2
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


class WhisperTranscriber:
    """Lazy-loading wrapper around faster-whisper with progress + cancellation."""

    def __init__(self, model_size="large-v3", device="auto", log=print):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Run: pip install faster-whisper"
            ) from exc
        self.device = _resolve_device(device)
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        log(f"Whisper device: {self.device} (compute_type={self.compute_type})")
        self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)

    def transcribe(self, audio_path, language=None, progress_cb=None, cancel=None):
        """Transcribe audio -> (list[Subtitle], detected_language, language_probability)."""
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        subs = []
        for seg in segments:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Cancelled during transcription.")
            text = seg.text.strip()
            if text:
                subs.append(Subtitle(len(subs) + 1, float(seg.start), float(seg.end), text))
            if progress_cb and duration > 0:
                progress_cb(min(float(seg.end) / duration, 1.0))
        return subs, info.language, float(getattr(info, "language_probability", 0.0) or 0.0)
