# Subtitle Studio

A desktop app (Python + CustomTkinter) that turns a **video or audio file** into:

- an extracted **audio track** (WAV or MP3),
- **subtitles** (`.srt` + `.vtt`) with automatic language detection,
- and optionally a **translated subtitle file** (timestamps preserved).

You can also load an existing `.srt` and translate it directly.

## Features

- Accepts common video formats (MP4, MKV, AVI, MOV, WebM, FLV, WMV, M4V, ...) and audio formats (MP3, WAV, M4A, FLAC, OGG, AAC, WMA, OPUS, ...)
- Audio-only extraction at original quality (WAV) or MP3
- Transcription with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2): fast on GPU, automatic language detection, VAD filtering, live progress bar
- Subtitle translation with two engines:
  - **Google Translate (online)** - zero extra downloads, needs internet
  - **NLLB-200 distilled 600M (offline)** - runs locally on your GPU/CPU, private
- 24 built-in languages, including English, Turkish, Spanish, Hindi, Urdu, French, Chinese (Simplified/Traditional), Japanese, Korean, Arabic, German, Russian, and more
- Background worker thread: the GUI never freezes, and long jobs can be cancelled
- No separate ffmpeg install needed (uses the `imageio-ffmpeg` bundled binary)

## Pipeline

```
video/audio file
      |
      |--(extract audio only)--> name_audio.wav / .mp3
      |
      +--> ffmpeg -> 16 kHz mono WAV -> faster-whisper -> name.<lang>.srt + .vtt
                                                              |
                                            (optional) translate -> name.<target>.srt
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

### Optional: offline translation (NLLB-200)

Uncomment the three lines at the bottom of `requirements.txt`, then re-run
`pip install -r requirements.txt`. The model (~2.4 GB) downloads from Hugging
Face on first use.

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
3. Press **Start**. Watch the progress bar and log; **Cancel** stops between segments/batches.
4. **Open output folder** jumps straight to the results.

### Output files (for input `lecture.mp4`)

| File | Content |
|---|---|
| `lecture_audio.wav` / `.mp3` | extracted audio track |
| `lecture.en.srt`, `lecture.en.vtt` | subtitles in the detected language |
| `lecture.tr.srt` | subtitles translated to the target language (e.g. Turkish) |

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

Translation preserves cue indices and timestamps exactly - only the text changes.

## Hardware fit (32 GB RAM / 16 GB VRAM)

`large-v3` in `float16` (~6-8 GB VRAM) plus NLLB-200 in `float16` (~1.5 GB)
fit comfortably in 16 GB VRAM, and 32 GB system RAM is plenty for long videos.
If you ever hit OOM, switch the model to `medium` or use CPU + `small`.

## Troubleshooting

- **"ffmpeg was not found"** - `pip install imageio-ffmpeg` (or install system ffmpeg).
- **Whisper fails to load on CUDA** - install the CUDA 12 runtime libs (see GPU notes), or set Device to `CPU`.
- **NLLB engine errors on start** - the optional deps are not installed; see the section above.
- **Google engine returns errors** - you are offline or rate-limited; wait a minute or switch to NLLB.
- **First run is slow** - model weights are downloading; subsequent runs start instantly.

## Project structure

```
subtitle_studio/
|-- app.py                 # CustomTkinter GUI (run this)
|-- core/
|   |-- languages.py       # one registry: Whisper / Google / NLLB codes
|   |-- media.py           # ffmpeg audio extraction, format detection
|   |-- subtitles.py       # SRT/VTT parse + write (dependency-free)
|   |-- transcriber.py     # faster-whisper wrapper (progress + cancel)
|   |-- translator.py      # Google + NLLB engines, SRT translation driver
|   `-- pipeline.py        # job orchestration (extract -> transcribe -> translate)
|-- requirements.txt
`-- README.md
```

## Notes and limitations

- Whisper's built-in `translate` task only translates *into English*, so this app
  uses dedicated translators to support many source/target pairs.
- Translation is cue-by-cue; very long cues are sent as-is (NLLB truncates at 512 tokens).
- The Google engine uses an unofficial endpoint - for heavy or sensitive use, prefer NLLB.
