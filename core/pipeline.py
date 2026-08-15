"""End-to-end job orchestration: extract audio -> transcribe -> translate."""
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .languages import google_code, name_for_whisper, whisper_code
from .media import extract_audio, extract_for_asr, is_audio, is_video
from .subtitles import write_srt, write_vtt
from .transcriber import WhisperTranscriber
from .translator import ENGINE_LABELS, make_engine, resolve_codes, translate_srt_file


@dataclass
class JobConfig:
    input_path: str
    output_dir: str
    save_audio: bool = False
    audio_format: str = "wav"               # "wav" | "mp3"
    transcribe: bool = True
    translate: bool = False
    source_language: Optional[str] = None   # Whisper code; None = auto-detect
    target_language: str = "English"        # display name from LANGUAGES
    translation_engine: str = "google"      # "google" | "nllb"
    whisper_model: str = "large-v3"
    device: str = "auto"                    # "auto" | "cuda" | "cpu"


def _check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise InterruptedError("Cancelled by user.")


def run_job(cfg, log=print, progress=None, cancel=None):
    """Run the configured job. Returns the list of files that were written."""
    progress = progress or (lambda frac, text="": None)
    inp = Path(cfg.input_path)
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = inp.stem
    ext = inp.suffix.lower()
    outputs: List[Path] = []

    if ext == ".srt":
        # Subtitle-only workflow: translate an existing SRT.
        if not cfg.translate:
            raise ValueError("An .srt file was loaded - enable 'Translate subtitles'.")
        srt_path = inp
        src_lang = cfg.source_language
        translate_base = 0.05
    else:
        if not (is_video(inp) or is_audio(inp)):
            raise ValueError(f"Unsupported file type: {ext or '(none)'}")
        if not (cfg.save_audio or cfg.transcribe or cfg.translate):
            raise ValueError("Nothing to do - enable at least one task.")

        if cfg.save_audio:
            progress(0.02, "Extracting audio...")
            log(f"Extracting audio ({cfg.audio_format})...")
            audio_path = extract_audio(
                inp, out_dir / f"{stem}_audio.{cfg.audio_format}", cfg.audio_format, log=log
            )
            outputs.append(audio_path)
            log(f"Saved audio: {audio_path.name}")

        srt_path = None
        src_lang = cfg.source_language
        translate_base = 0.78

        if cfg.transcribe or cfg.translate:
            _check_cancel(cancel)
            progress(0.05, f"Loading Whisper model '{cfg.whisper_model}'...")
            log(f"Loading Whisper model '{cfg.whisper_model}' (downloads on first run)...")
            transcriber = WhisperTranscriber(cfg.whisper_model, cfg.device, log=log)
            with tempfile.TemporaryDirectory() as tmp:
                wav = extract_for_asr(inp, tmp)
                log("Transcribing...")
                subs, lang, prob = transcriber.transcribe(
                    wav,
                    language=cfg.source_language,
                    progress_cb=lambda f: progress(0.05 + 0.70 * f, "Transcribing..."),
                    cancel=cancel,
                )
            if not subs:
                raise RuntimeError("No speech was detected in the input.")
            src_lang = lang
            lang_name = name_for_whisper(lang) or lang
            if cfg.source_language:
                log(f"Language: {lang_name} [{lang}]")
            else:
                log(f"Detected language: {lang_name} [{lang}] (confidence {prob:.0%})")
            srt_path = out_dir / f"{stem}.{lang}.srt"
            vtt_path = out_dir / f"{stem}.{lang}.vtt"
            write_srt(subs, srt_path)
            write_vtt(subs, vtt_path)
            outputs += [srt_path, vtt_path]
            log(f"Saved subtitles: {srt_path.name}, {vtt_path.name}")

    if cfg.translate:
        _check_cancel(cancel)
        if src_lang and src_lang == whisper_code(cfg.target_language):
            log("Note: target language matches the source - output will mirror the input.")
        dst = out_dir / f"{stem}.{google_code(cfg.target_language)}.srt"
        engine = make_engine(cfg.translation_engine, cfg.device, log=log)
        src_code, tgt_code = resolve_codes(cfg.translation_engine, src_lang, cfg.target_language)
        log(f"Translating to {cfg.target_language} via {ENGINE_LABELS[cfg.translation_engine]}...")
        translate_srt_file(
            srt_path, dst, engine, src_code, tgt_code,
            log=log, cancel=cancel,
            progress_cb=lambda f: progress(
                translate_base + (1.0 - translate_base) * f, "Translating..."
            ),
        )
        outputs.append(dst)
        log(f"Saved translated subtitles: {dst.name}")

    progress(1.0, "Done")
    return outputs
