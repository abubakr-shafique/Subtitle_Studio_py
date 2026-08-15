# Subtitle Studio

A desktop app built with Python and CustomTkinter that converts video or audio into:

- extracted audio (`.wav` or `.mp3`),
- subtitles (`.srt` and `.vtt`) with automatic language detection,
- optionally translated subtitles,
- optionally translated/dubbed audio aligned to the original timeline.

It also accepts an existing `.srt` file for translation and Edge-TTS dubbing.

## Features

- Video: MP4, MKV, AVI, MOV, WebM, FLV, WMV, M4V, MPEG, MPG, 3GP.
- Audio: MP3, WAV, M4A, FLAC, OGG, AAC, WMA, OPUS, AIFF.
- Automatic transcription and language detection with faster-whisper.
- SRT and VTT generation.
- Translation using Google Translate or local NLLB-200.
- Translated audio using Edge TTS or local Coqui XTTS v2.
- Batch-level translation failsafe: failed batches are retried once, then skipped while the original text is preserved.
- TTS failsafe: a failed subtitle cue is left silent while the remaining cues continue.
- Background processing, progress reporting, logging, and cancellation.

## Installation

Python 3.10+ is required. For the tested XTTS setup, use Python 3.12.

### Conda environment

Windows Anaconda Prompt:

```bat
conda create --name Subtitle_Studio_py python=3.12
conda activate Subtitle_Studio_py
```

### Install PyTorch first

For an NVIDIA GPU, install the PyTorch build matching your CUDA setup. The following is a CUDA 12.6 example:

```bat
python -m pip install torch torchvision torchaudio torchcodec --index-url https://download.pytorch.org/whl/cu126
```

For CPU-only installation:

```bat
python -m pip install torch torchvision torchaudio torchcodec
```

### Install the application dependencies

```bat
python -m pip install -r requirements.txt
```

Always use `python -m pip` rather than plain `pip`; this ensures packages are installed into the same interpreter that launches the app.

## Verified Coqui / Transformers setup

The current Coqui XTTS code imports:

```python
from transformers.pytorch_utils import isin_mps_friendly
```

Therefore, do **not** use the older `transformers==4.46.2` pin with this setup. Use a recent Transformers 4.x release:

```bat
python -m pip install -U "transformers>=4.57,<5"
python -m pip install -U "coqui-tts>=0.27.4"
```

Verify the environment before launching the GUI:

```bat
python -c "import sys; print(sys.executable)"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "from transformers.pytorch_utils import isin_mps_friendly; print('Transformers API OK')"
python -c "from TTS.api import TTS; print('Coqui API import OK')"
python -m pip check
```

All commands should use the same environment. 

The package is installed as `coqui-tts`, but its Python import is intentionally:

```python
from TTS.api import TTS
```

Do not install the obsolete package with `pip install TTS`.

## Launch

```bat
python app.py
```

1. Choose a video, audio, or `.srt` input.
2. Choose one or more tasks.
3. Enable **Translate subtitles** if translation is needed.
4. Enable **Generate translated audio** for dubbing.
5. Choose the translation and TTS engines.
6. Click **Start**.

The GUI's TTS option automatically requires translated subtitles. XTTS voice cloning requires the original video/audio because it uses a short source-speaker reference. For an `.srt` input or Urdu target, use Edge TTS.

## Output examples

For `lecture.mp4` translated to Turkish:

```text
lecture_audio.wav
lecture.en.srt
lecture.en.vtt
lecture.tr.srt
lecture.tr.dub.wav
```

The dubbed WAV is aligned to the original subtitle timestamps. If a translated cue is longer than the available gap, it is faded and truncated at the next cue to prevent overlapping speech.

To mux the translated audio back into the original video:

```bat
ffmpeg -i lecture.mp4 -i lecture.tr.dub.wav -map 0:v -map 1:a -c:v copy -shortest lecture_tr.mp4
```

## Translation failsafe

Translation is processed in batches of 32 cues:

1. Try the batch.
2. If it fails, wait briefly and retry once.
3. If it fails again, preserve the original text for that batch.
4. Continue with the remaining batches.

The log reports failed batches and the number of cues that remained in the source language.

## TTS engines

| Engine | Internet | Voice | Notes |
|---|---:|---|---|
| Edge TTS | Required | Microsoft neural voice | Supports all built-in languages, including Urdu. |
| XTTS v2 | No after model download | Cloned source voice | Supports 17 languages; Urdu is not supported. Requires the original media. |

XTTS v2 is under the Coqui Public Model License (CPML), which restricts use to permitted non-commercial purposes. Review the model license before redistribution or commercial use.

## Hardware

The target configuration is 32 GB RAM and 16 GB VRAM:

- `large-v3` Whisper: approximately 6–8 GB VRAM.
- NLLB-200: approximately 1.5–2 GB VRAM in half precision.
- XTTS v2: typically a few GB of VRAM depending on PyTorch/model configuration.

If memory is insufficient, select Whisper `medium` or `small`, or choose CPU mode.

## Troubleshooting

### Coqui API import error

Run:

```bat
python -c "from TTS.api import TTS; print('Coqui API import OK')"
```

If it fails with `isin_mps_friendly`, update Transformers:

```bat
python -m pip install -U "transformers>=4.57,<5"
```

If it fails because of PyTorch or audio backends:

```bat
python -m pip install -U torch torchaudio torchcodec
```

Then verify:

```bat
python -m pip check
python -c "from TTS.api import TTS; print('Coqui API import OK')"
```

If the package is installed but the GUI cannot import it, print the interpreter used by the GUI and compare it with the installation environment:

```bat
python -c "import sys; print(sys.executable)"
python -m pip --version
```

Also make sure the project does not contain files/folders named `TTS.py`, `TTS`, `transformers.py`, or `transformers`, because these can shadow installed packages.

### CUDA errors

Set the GUI device to **CPU** to test the software path. If CPU works but CUDA fails, reinstall the PyTorch wheel matching the installed NVIDIA driver/CUDA runtime.

### First-run downloads

Whisper, NLLB, and XTTS download model weights on first use. Keep the internet connection active during the first initialization of each model.

## Project structure

```text
subtitle_studio/
|-- app.py
|-- core/
|   |-- languages.py
|   |-- media.py
|   |-- subtitles.py
|   |-- transcriber.py
|   |-- translator.py
|   |-- tts.py
|   `-- pipeline.py
|-- requirements.txt
`-- README.md
```

## Limitations

- Translation is cue-by-cue.
- NLLB may truncate very long cues at its configured maximum length.
- Google Translate and Edge TTS require internet access and may be rate-limited.
- Dubbing is timestamp-aligned but not lip-synced.
- XTTS voice cloning is subject to the model license and should only be used with appropriate authorization for the reference voice.
