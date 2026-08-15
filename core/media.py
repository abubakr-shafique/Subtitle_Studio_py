"""Media type detection, duration probing, and audio extraction via ffmpeg."""
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".mpeg", ".mpg", ".3gp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus", ".aiff"}
SUBTITLE_EXTS = {".srt"}

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


@lru_cache(maxsize=1)
def ffmpeg_exe():
    """Locate an ffmpeg binary: bundled (imageio-ffmpeg) first, then system PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise RuntimeError(
        "ffmpeg was not found. Install it with `pip install imageio-ffmpeg` "
        "or install a system-wide ffmpeg."
    )


def _ext(path):
    return Path(path).suffix.lower()


def is_video(path):
    return _ext(path) in VIDEO_EXTS


def is_audio(path):
    return _ext(path) in AUDIO_EXTS


def is_subtitle(path):
    return _ext(path) in SUBTITLE_EXTS


def media_duration(path):
    """Container duration in seconds, parsed from `ffmpeg -i` output."""
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                          capture_output=True, text=True)
    m = _DURATION_RE.search(proc.stderr or "")
    if not m:
        raise RuntimeError(f"Could not determine the duration of {Path(path).name}.")
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def extract_audio(input_path, output_path, fmt="wav", log=print):
    """Extract the audio track at (near-)original quality. Returns the actual output path."""
    input_path, output_path = Path(input_path), Path(output_path)
    fmt = fmt.lower().lstrip(".")
    if output_path.resolve() == input_path.resolve():
        output_path = output_path.with_name(output_path.stem + "_out" + output_path.suffix)

    def _run(codec_args, out):
        cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
               "-i", str(input_path), "-vn", *codec_args, str(out)]
        return subprocess.run(cmd, capture_output=True, text=True)

    if fmt == "mp3":
        proc = _run(["-acodec", "libmp3lame", "-q:a", "2"], output_path)
        if proc.returncode != 0:
            log("MP3 encoder unavailable in this ffmpeg build - falling back to WAV.")
            output_path = output_path.with_suffix(".wav")
            fmt = "wav"
    if fmt == "wav":
        proc = _run(["-acodec", "pcm_s16le"], output_path)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{proc.stderr[-1500:]}")
    return output_path


def extract_for_asr(input_path, work_dir):
    """Decode any media file to a 16 kHz mono WAV - the ideal input for Whisper."""
    out = Path(work_dir) / "asr_input_16k.wav"
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(input_path), "-vn", "-ac", "1", "-ar", "16000",
           "-acodec", "pcm_s16le", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode the input:\n{proc.stderr[-1500:]}")
    return out


def normalize_wav(input_path, output_path, sample_rate=24000):
    """Convert any audio clip to mono PCM s16le WAV at the given sample rate."""
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(input_path), "-ac", "1", "-ar", str(sample_rate),
           "-acodec", "pcm_s16le", str(output_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not convert audio:\n{proc.stderr[-1500:]}")
    return Path(output_path)
