"""Subtitle translation: Google Translate (online) or NLLB-200 (offline, local)."""
from pathlib import Path

from .languages import google_code, nllb_code, nllb_for_whisper
from .subtitles import compose_srt, parse_srt

ENGINE_LABELS = {
    "google": "Google Translate (online)",
    "nllb": "NLLB-200 (offline, local model)",
}
BATCH_SIZE = 32


def resolve_codes(engine_key, source_whisper, target_name):
    """Map (detected Whisper code or None, target display name) to engine-specific codes."""
    if engine_key == "google":
        return (source_whisper or "auto"), google_code(target_name)
    if engine_key == "nllb":
        if not source_whisper:
            raise ValueError(
                "The offline NLLB engine needs a known source language. For media input "
                "it is filled in automatically after detection; for .srt input, pick the "
                "source language in the drop-down."
            )
        return nllb_for_whisper(source_whisper), nllb_code(target_name)
    raise ValueError(f"Unknown translation engine: {engine_key}")


def make_engine(engine_key, device="auto", log=print):
    if engine_key == "google":
        return GoogleEngine(log)
    if engine_key == "nllb":
        return NLLBEngine(device, log)
    raise ValueError(f"Unknown translation engine: {engine_key}")


class GoogleEngine:
    """Free Google Translate endpoint via deep-translator. Requires internet."""

    def __init__(self, log=print):
        try:
            from deep_translator import GoogleTranslator
        except ImportError as exc:
            raise RuntimeError(
                "The Google engine needs deep-translator. Run: pip install deep-translator"
            ) from exc
        self._translator_cls = GoogleTranslator
        self.log = log

    def translate(self, texts, source, target):
        translator = self._translator_cls(source=source, target=target)
        try:
            return [t or "" for t in translator.translate_batch(list(texts))]
        except Exception as exc:
            self.log(f"Batch request failed ({exc}); retrying line by line...")
            return [(translator.translate(t) or "") for t in texts]


class NLLBEngine:
    """Meta NLLB-200 distilled 600M running locally via Hugging Face transformers."""

    MODEL_NAME = "facebook/nllb-200-distilled-600M"

    def __init__(self, device="auto", log=print):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
        except ImportError as exc:
            raise RuntimeError(
                "The NLLB engine needs extra packages. Run: "
                "pip install torch transformers sentencepiece"
            ) from exc
        use_cuda = device in ("auto", "cuda") and torch.cuda.is_available()
        log(f"Loading {self.MODEL_NAME} on {'GPU' if use_cuda else 'CPU'} "
            f"(downloads ~2.4 GB on first run)...")
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
        )
        self._pipe = pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer,
            device=0 if use_cuda else -1,
            batch_size=16,
        )

    def translate(self, texts, source, target):
        results = self._pipe(list(texts), src_lang=source, tgt_lang=target, max_length=512)
        return [r["translation_text"] for r in results]


def translate_srt_file(src_path, dst_path, engine, src_code, tgt_code,
                       log=print, cancel=None, progress_cb=None):
    """Translate every cue of an SRT file, preserving indices and timestamps."""
    subs = parse_srt(Path(src_path).read_text(encoding="utf-8-sig"))
    if not subs:
        raise ValueError(f"No subtitle cues could be parsed from {Path(src_path).name}.")
    texts = [s.text for s in subs]
    total = len(texts)
    done = 0
    for start in range(0, total, BATCH_SIZE):
        if cancel is not None and cancel.is_set():
            raise InterruptedError("Cancelled during translation.")
        chunk = texts[start:start + BATCH_SIZE]
        idxs = [i for i, t in enumerate(chunk) if t.strip()]
        payload = [chunk[i] for i in idxs]
        if payload:
            translated = engine.translate(payload, src_code, tgt_code)
            for i, t in zip(idxs, translated):
                chunk[i] = t.strip()
        for i, t in enumerate(chunk):
            subs[start + i] = subs[start + i]._replace(text=t)
        done += len(chunk)
        if progress_cb:
            progress_cb(done / total)
        log(f"Translated {done}/{total} cues")
    dst_path = Path(dst_path)
    dst_path.write_text(compose_srt(subs), encoding="utf-8")
    return dst_path
