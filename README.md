# Subtitle Studio

A desktop app (Python + CustomTkinter) that turns a **video or audio file** into:

- an extracted **audio track** (WAV or MP3),
- **subtitles** (`.srt` + `.vtt`) with automatic language detection,
- optionally a **translated subtitle file** (timestamps preserved),
- and optionally a **translated audio track** (TTS dubbing aligned to the original timeline).

You can also load an existing `.srt` and translate / dub it directly.

## Features

- Accepts common video formats (MP4, MKV, AVI, MOV, WebM, FLV, WMV, M4V, ...) and audio formats (MP3, WAV, M4A, FLAC, OGG, AAC, WMA, OPUS, ...)
- Audio-only extraction at original quality (WAV) or MP3
- Transcription with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2): fast on GPU, automatic language detection, VAD filtering, live progress bar
- Subtitle translation with two engines:
  - **Google Translate (online)** - zero extra downloads, needs internet
  - **NLLB-200 distilled 600M (offline)** - runs locally on your GPU/CPU, private
- **Translated audio (TTS dubbing)** with two engines:
  - **Edge TTS (online)** - neural voices for every built-in language, including Urdu and Hindi
  - **XTTS v2 (offline)** - clones the original speaker's voice from a 30 s reference (17 languages, no Urdu)
- 24 built-in languages, including English, Turkish, Spanish, Hindi, Urdu, French, Chinese (Simplified/Traditional), Japanese, Korean, Arabic, German, Russian, and more
- Resilient by design: a failed translation batch is retried once, then skipped (original text kept) while the rest of the file continues; a failed TTS cue is left silent. Nothing aborts mid-file.
- Background worker thread: the GUI never freezes, and long jobs can be cancelled
- No separate ffmpeg install needed (uses the `imageio-ffmpeg` bundled binary)

## Pipeline

```
video/audio file
      |
      |--(extract audio only)----------------> name_audio.wav / .mp3
      |
      +--> ffmpeg 16 kHz WAV -> faster-whisper -> name.<lang>.srt + .vtt
                                                              |
                                            (optional) translate -> name.<target>.srt
                                                              |
                                            (optional) TTS dub  -> name.<target>.dub.wav
```

## Installation

Requires **Python 3.10+** (3.10-3.12 recommended).

```bash
git clone <your-repo> Subtitle_Studio_py   # or just unzip the project
cd Subtitle_Studio_py
conda create --name Subtitle_Studio_py python=3.10
conda activate Subtitle_Studio_py      # Windows
pip install -r requirements.txt
```

### Optional extras

Uncomment the relevant lines at the bottom of `requirements.txt`, then re-run
`pip install -r requirements.txt`:

- **NLLB-200** (offline translation): torch + transformers + sentencepiece. ~2.4 GB model download on first use.
- **Edge TTS** (online translated audio): `edge-tts`. Small, pure Python.
- **XTTS v2** (offline translated audio with voice cloning): `coqui-tts`. ~1.8 GB model download on first use. Note the model is under the **CPML non-commercial license**.

### GPU notes (CUDA)

- With an NVIDIA GPU, faster-whisper runs on CUDA automatically (`Device: Auto`).
  On Linux you may need the CUDA runtime libraries:
  `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`
  (see the faster-whisper docs for details). On Windows, install the CUDA 12.x toolkit.
- No GPU? Everything still works on CPU (Whisper falls back to `int8`); pick a
  smaller model such as `small` or `base` for usable speed.

## Usage

```bash
python app.py
```

1. **Browse...** - pick a video, audio, or `.srt` file. The output folder defaults to the input's folder.
2. Tick any combination of tasks:
   - **Extract audio only** - choose `wav` (lossless) or `mp3`.
   - **Generate subtitles** - pick the Whisper model and device.
   - **Translate subtitles** - pick source (`Auto-detect` uses Whisper's detected language), target language, and engine.
   - **Generate translated audio** - pick the TTS engine (requires translation to be on; the app offers to enable it for you).
3. Press **Start**. Watch the progress bar and log; **Cancel** stops between segments/batches.
4. **Open output folder** jumps straight to the results.

### Output files (for input `lecture.mp4`, target Turkish)

| File | Content |
|---|---|
| `lecture_audio.wav` / `.mp3` | extracted audio track |
| `lecture.en.srt`, `lecture.en.vtt` | subtitles in the detected language |
| `lecture.tr.srt` | translated subtitles (timestamps preserved) |
| `lecture.tr.dub.wav` | translated speech, aligned to the original timeline (24 kHz WAV) |

## How the dubbing syncs

Each translated cue is synthesized and placed at its **original start time** on a
silent canvas the length of the source media. If a synthesized clip is longer
than the gap before the next cue, it is faded out (40 ms) and truncated so cues
never talk over each other; short clips leave natural silence. The result is a
WAV you can mux back over the video, e.g.:

```bash
ffmpeg -i lecture.mp4 -i lecture.tr.dub.wav -map 0:v -map 1:a -c:v copy -shortest lecture_tr.mp4
```

## Choosing a Whisper model

Approximate VRAM with `float16` on GPU (CPU uses `int8` and roughly half the RAM):

| Model | ~VRAM | Relative speed | Notes |
|---|---|---|---|
| `tiny` | ~1 GB | ~10x realtime | drafts only |
| `base` | ~1 GB | ~7x | quick checks |
| `small` | ~2 GB | ~4x | decent quality |
| `medium` | ~5 GB | ~2x | good quality |
| `large-v3` | ~6-8 GB | ~1x | best accuracy - recommended on your 16 GB GPU |

Models download from Hugging Face on first use (`large-v3` is ~3 GB).

## Translation engines

| Engine | Internet | Extra install | Notes |
|---|---|---|---|
| Google Translate | required | none | fast, free endpoint; unofficial API, very long jobs may be rate-limited |
| NLLB-200 (offline) | no | torch + transformers + sentencepiece | private, runs on your GPU; ~2.4 GB model download once |

**Failsafe:** translation runs in batches of 32 cues. A failed batch is retried
once after a short pause; if it still fails, those cues keep their original text
and the job continues. The log reports how many cues (if any) were left
untranslated, so you can re-run just to patch them.

## TTS engines (translated audio)

| Engine | Internet | Extra install | Languages | Notes |
|---|---|---|---|---|
| Edge TTS | required | `edge-tts` | all 24 built-in (incl. Urdu, Hindi) | neural voices, one default voice per language - edit `EDGE_VOICES` in `core/tts.py` to change |
| XTTS v2 | no | `coqui-tts` | 17 (no Urdu) | clones the original speaker from a 30 s reference auto-cut from your file; CPML non-commercial license |

**Failsafe:** a cue that fails to synthesize is left silent and dubbing
continues; the log reports how many cues were spoken vs skipped.

## Hardware fit (32 GB RAM / 16 GB VRAM)

`large-v3` in `float16` (~6-8 GB VRAM), NLLB-200 in `float16` (~1.5 GB), and
XTTS v2 (~2 GB) each fit comfortably - even two at a time - in 16 GB VRAM, and
32 GB system RAM is plenty for long videos. If you ever hit OOM, switch the
Whisper model to `medium` or use CPU + `small`.

## Troubleshooting

- **"ffmpeg was not found"** - `pip install imageio-ffmpeg` (or install system ffmpeg).
- **Whisper fails to load on CUDA** - install the CUDA 12 runtime libs (see GPU notes), or set Device to `CPU`.
- **NLLB engine errors on start** - the optional deps are not installed; see Optional extras.
- **Google engine leaves cues untranslated** - you were offline or rate-limited; the failsafe kept the original text. Wait a minute and re-run, or switch to NLLB.
- **"XTTS v2 does not support ..."** - that target (e.g. Urdu) has no XTTS voice; use Edge TTS for it.
- **"XTTS voice cloning needs the original media"** - you loaded an `.srt`; voice cloning needs the source audio/video. Use Edge TTS for `.srt` input.
- **First run is slow** - model weights are downloading; subsequent runs start instantly.

## Project structure

```
subtitle_studio/
|-- app.py                 # CustomTkinter GUI (run this)
|-- core/
|   |-- languages.py       # one registry: Whisper / Google / NLLB codes
|   |-- media.py           # ffmpeg audio extraction, duration probe, WAV normalize
|   |-- subtitles.py       # SRT/VTT parse + write (dependency-free)
|   |-- transcriber.py     # faster-whisper wrapper (progress + cancel)
|   |-- translator.py      # Google + NLLB engines, batch failsafe, SRT driver
|   |-- tts.py             # Edge TTS + XTTS engines, timeline dubbing mixer
|   `-- pipeline.py        # job orchestration (extract -> transcribe -> translate -> dub)
|-- requirements.txt
`-- README.md
```

## Notes and limitations

- Whisper's built-in `translate` task only translates *into English*, so this app
  uses dedicated translators to support many source/target pairs.
- Translation is cue-by-cue; very long cues are sent as-is (NLLB truncates at 512 tokens).
- The Google and Edge engines use unofficial free endpoints - for heavy, sensitive,
  or commercial use, prefer the offline engines (and respect the XTTS CPML license).
- Dubbing is cue-aligned, not lip-synced; translated speech that is much longer
  than the original cue gets truncated at the next cue boundary.
