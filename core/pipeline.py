"""End-to-end job orchestration: extract audio -> transcribe -> translate -> dub."""
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .languages import google_code, name_for_whisper, whisper_code
from .media import extract_audio, extract_for_asr, is_audio, is_video, media_duration
from .subtitles import write_srt, write_vtt
from .transcriber import WhisperTranscriber
from .translator import ENGINE_LABELS, make_engine, resolve_codes, translate_srt_file
from .tts import TTS_ENGINE_LABELS, dub_srt, make_tts_engine, trim_wav


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
    generate_audio: bool = False            # synthesize the translation into audio
    tts_engine: str = "edge"                # "edge" | "xtts"
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
    tgt_gcode = google_code(cfg.target_language)

    if ext == ".srt":
        # Subtitle-only workflow: translate (and optionally dub) an existing SRT.
        if not (cfg.translate or cfg.generate_audio):
            raise ValueError("An .srt file was loaded - enable 'Translate subtitles' and/or 'Translated audio'.")
        if cfg.generate_audio and not cfg.translate:
            raise ValueError("Translated audio requires 'Translate subtitles' to be enabled.")
        srt_path = inp
        src_lang = cfg.source_language
        duration = None
        reference_wav = None
        translate_base = 0.05
    else:
        if not (is_video(inp) or is_audio(inp)):
            raise ValueError(f"Unsupported file type: {ext or '(none)'}")
        if not (cfg.save_audio or cfg.transcribe or cfg.translate or cfg.generate_audio):
            raise ValueError("Nothing to do - enable at least one task.")
        if cfg.generate_audio and not cfg.translate:
            raise ValueError("Translated audio requires 'Translate subtitles' to be enabled.")
        duration = media_duration(inp)

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
        reference_wav = None
        need_asr = cfg.transcribe or cfg.translate or cfg.generate_audio
        translate_base = 0.62 if cfg.generate_audio else 0.78

        if need_asr:
            _check_cancel(cancel)
            progress(0.05, f"Loading Whisper model '{cfg.whisper_model}'...")
            log(f"Loading Whisper model '{cfg.whisper_model}' (downloads on first run)...")
            transcriber = WhisperTranscriber(cfg.whisper_model, cfg.device, log=log)
            with tempfile.TemporaryDirectory() as tmp:
                wav = extract_for_asr(inp, tmp)
                if cfg.generate_audio and cfg.tts_engine == "xtts":
                    reference_wav = trim_wav(wav, Path(tmp) / "xtts_reference.wav", 30)
                    log("Prepared a 30 s voice reference for XTTS cloning.")
                log("Transcribing...")
                subs, lang, prob = transcriber.transcribe(
                    wav,
                    language=cfg.source_language,
                    progress_cb=lambda f: progress(0.05 + 0.55 * f, "Transcribing..."),
                    cancel=cancel,
                )
                if cfg.generate_audio and cfg.tts_engine == "xtts":
                    ref_out = Path(tempfile.mkdtemp(prefix="subtitle_studio_ref_")) / "xtts_reference.wav"
                    ref_out.write_bytes(Path(reference_wav).read_bytes())
                    reference_wav = ref_out
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

    translated_path = None
    if cfg.translate or cfg.generate_audio:
        _check_cancel(cancel)
        if src_lang and src_lang == whisper_code(cfg.target_language):
            log("Note: target language matches the source - output will mirror the input.")
        translated_path = out_dir / f"{stem}.{tgt_gcode}.srt"
        engine = make_engine(cfg.translation_engine, cfg.device, log=log)
        src_code, tgt_code = resolve_codes(cfg.translation_engine, src_lang, cfg.target_language)
        log(f"Translating to {cfg.target_language} via {ENGINE_LABELS[cfg.translation_engine]}...")
        translate_srt_file(
            srt_path, translated_path, engine, src_code, tgt_code,
            log=log, cancel=cancel,
            progress_cb=lambda f: progress(
                translate_base + (0.92 - translate_base) * f, "Translating..."
            ),
        )
        outputs.append(translated_path)
        log(f"Saved translated subtitles: {translated_path.name}")

    if cfg.generate_audio:
        _check_cancel(cancel)
        out_wav = out_dir / f"{stem}.{tgt_gcode}.dub.wav"
        log(f"Generating translated audio via {TTS_ENGINE_LABELS[cfg.tts_engine]}...")
        tts = make_tts_engine(cfg.tts_engine, cfg.device, log=log, reference_wav=reference_wav)
        tts.check_language(tgt_gcode)
        dub_srt(
            translated_path, out_wav, tts, tgt_gcode, duration,
            log=log, cancel=cancel,
            progress_cb=lambda f: progress(0.92 + 0.08 * f, "Synthesizing audio..."),
        )
        outputs.append(out_wav)
        log(f"Saved translated audio: {out_wav.name}")

    progress(1.0, "Done")
    return outputs
