"""Subtitle Studio - desktop GUI (CustomTkinter).

Turn a video or audio file into extracted audio, SRT/VTT subtitles, and
optionally translated subtitles plus translated (dubbed) audio. Heavy work
runs on a background thread so the UI stays responsive; progress and logs
arrive through a queue.
"""
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from core.languages import LANGUAGES, whisper_code
from core.media import AUDIO_EXTS, SUBTITLE_EXTS, VIDEO_EXTS
from core.pipeline import JobConfig, run_job
from core.translator import ENGINE_LABELS
from core.tts import TTS_ENGINE_LABELS

APP_TITLE = "Subtitle Studio"
MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]
DEVICE_CHOICES = {"Auto": "auto", "CUDA (GPU)": "cuda", "CPU": "cpu"}
AUDIO_FORMATS = ["wav", "mp3"]
LANG_NAMES = list(LANGUAGES.keys())
SOURCE_CHOICES = ["Auto-detect"] + LANG_NAMES
ENGINE_NAMES = list(ENGINE_LABELS.values())
ENGINE_KEYS = {v: k for k, v in ENGINE_LABELS.items()}
TTS_ENGINE_NAMES = list(TTS_ENGINE_LABELS.values())
TTS_ENGINE_KEYS = {v: k for k, v in TTS_ENGINE_LABELS.items()}


def _patterns(exts):
    return " ".join(f"*{e}" for e in sorted(exts))


FILETYPES = [
    ("Media and subtitles", _patterns(VIDEO_EXTS | AUDIO_EXTS | SUBTITLE_EXTS)),
    ("Video files", _patterns(VIDEO_EXTS)),
    ("Audio files", _patterns(AUDIO_EXTS)),
    ("Subtitle files", "*.srt"),
    ("All files", "*.*"),
]


class SubtitleStudioApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1000x880")
        self.minsize(900, 800)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.queue = queue.Queue()
        self.worker = None
        self.cancel_event = threading.Event()
        self.input_path = ctk.StringVar()
        self.output_dir = ctk.StringVar()

        self._build_ui()
        self._refresh_states()
        self.after(100, self._poll_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- UI construction ----------------
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        files = ctk.CTkFrame(self)
        files.grid(row=0, column=0, padx=14, pady=(14, 6), sticky="ew")
        files.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(files, text="Input file:").grid(row=0, column=0, padx=(12, 6), pady=8, sticky="w")
        ctk.CTkEntry(files, textvariable=self.input_path, state="readonly").grid(
            row=0, column=1, padx=6, pady=8, sticky="ew")
        ctk.CTkButton(files, text="Browse...", width=110, command=self._browse_input).grid(
            row=0, column=2, padx=(6, 12), pady=8)
        ctk.CTkLabel(files, text="Output folder:").grid(row=1, column=0, padx=(12, 6), pady=(0, 10), sticky="w")
        ctk.CTkEntry(files, textvariable=self.output_dir, state="readonly").grid(
            row=1, column=1, padx=6, pady=(0, 10), sticky="ew")
        ctk.CTkButton(files, text="Browse...", width=110, command=self._browse_output).grid(
            row=1, column=2, padx=(6, 12), pady=(0, 10))

        opts = ctk.CTkFrame(self)
        opts.grid(row=1, column=0, padx=14, pady=6, sticky="ew")
        opts.grid_columnconfigure((0, 1), weight=1)

        audio_box = ctk.CTkFrame(opts)
        audio_box.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.chk_audio = ctk.CTkCheckBox(audio_box, text="Extract audio only", command=self._refresh_states)
        self.chk_audio.grid(row=0, column=0, columnspan=2, padx=12, pady=(12, 4), sticky="w")
        ctk.CTkLabel(audio_box, text="Audio format:").grid(row=1, column=0, padx=12, pady=(0, 12), sticky="w")
        self.audio_fmt_menu = ctk.CTkOptionMenu(audio_box, values=AUDIO_FORMATS, width=110)
        self.audio_fmt_menu.grid(row=1, column=1, padx=12, pady=(0, 12), sticky="w")
        self.audio_fmt_menu.set("wav")

        subs_box = ctk.CTkFrame(opts)
        subs_box.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.chk_subs = ctk.CTkCheckBox(subs_box, text="Generate subtitles (SRT + VTT)", command=self._refresh_states)
        self.chk_subs.grid(row=0, column=0, columnspan=2, padx=12, pady=(12, 4), sticky="w")
        ctk.CTkLabel(subs_box, text="Whisper model:").grid(row=1, column=0, padx=12, sticky="w")
        self.model_menu = ctk.CTkOptionMenu(subs_box, values=MODEL_SIZES, width=160)
        self.model_menu.grid(row=1, column=1, padx=12, sticky="w")
        self.model_menu.set("large-v3")
        ctk.CTkLabel(subs_box, text="Device:").grid(row=2, column=0, padx=12, pady=(0, 12), sticky="w")
        self.device_menu = ctk.CTkOptionMenu(subs_box, values=list(DEVICE_CHOICES.keys()), width=160)
        self.device_menu.grid(row=2, column=1, padx=12, pady=(0, 12), sticky="w")
        self.device_menu.set("Auto")

        trans_box = ctk.CTkFrame(opts)
        trans_box.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 6), sticky="nsew")
        self.chk_translate = ctk.CTkCheckBox(trans_box, text="Translate subtitles", command=self._refresh_states)
        self.chk_translate.grid(row=0, column=0, columnspan=6, padx=12, pady=(12, 4), sticky="w")
        ctk.CTkLabel(trans_box, text="Source language:").grid(row=1, column=0, padx=12, pady=(0, 12), sticky="w")
        self.source_menu = ctk.CTkOptionMenu(trans_box, values=SOURCE_CHOICES, width=190)
        self.source_menu.grid(row=1, column=1, padx=8, pady=(0, 12), sticky="w")
        self.source_menu.set("Auto-detect")
        ctk.CTkLabel(trans_box, text="Target language:").grid(row=1, column=2, padx=12, pady=(0, 12), sticky="w")
        self.target_menu = ctk.CTkOptionMenu(trans_box, values=LANG_NAMES, width=190)
        self.target_menu.grid(row=1, column=3, padx=8, pady=(0, 12), sticky="w")
        self.target_menu.set("English")
        ctk.CTkLabel(trans_box, text="Engine:").grid(row=1, column=4, padx=12, pady=(0, 12), sticky="w")
        self.engine_menu = ctk.CTkOptionMenu(trans_box, values=ENGINE_NAMES, width=240)
        self.engine_menu.grid(row=1, column=5, padx=(8, 12), pady=(0, 12), sticky="w")
        self.engine_menu.set(ENGINE_NAMES[0])

        tts_box = ctk.CTkFrame(opts)
        tts_box.grid(row=2, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="nsew")
        self.chk_tts = ctk.CTkCheckBox(
            tts_box, text="Generate translated audio (TTS dub)", command=self._refresh_states
        )
        self.chk_tts.grid(row=0, column=0, columnspan=3, padx=12, pady=(12, 4), sticky="w")
        ctk.CTkLabel(tts_box, text="TTS engine:").grid(row=1, column=0, padx=12, pady=(0, 12), sticky="w")
        self.tts_menu = ctk.CTkOptionMenu(tts_box, values=TTS_ENGINE_NAMES, width=270)
        self.tts_menu.grid(row=1, column=1, padx=8, pady=(0, 12), sticky="w")
        self.tts_menu.set(TTS_ENGINE_NAMES[0])
        ctk.CTkLabel(
            tts_box,
            text="Requires 'Translate subtitles'. XTTS clones the original speaker (no Urdu).",
            text_color="gray", font=("", 11),
        ).grid(row=1, column=2, padx=12, pady=(0, 12), sticky="w")

        actions = ctk.CTkFrame(self)
        actions.grid(row=2, column=0, padx=14, pady=6, sticky="ew")
        actions.grid_columnconfigure(3, weight=1)
        self.start_btn = ctk.CTkButton(actions, text="Start", width=130, command=self._start)
        self.start_btn.grid(row=0, column=0, padx=(12, 6), pady=10)
        self.cancel_btn = ctk.CTkButton(actions, text="Cancel", width=100, fg_color="#8a2f2f",
                                        hover_color="#6e2424", command=self._cancel, state="disabled")
        self.cancel_btn.grid(row=0, column=1, padx=6, pady=10)
        self.open_btn = ctk.CTkButton(actions, text="Open output folder", width=160, command=self._open_output)
        self.open_btn.grid(row=0, column=2, padx=6, pady=10)
        self.progress = ctk.CTkProgressBar(actions)
        self.progress.grid(row=0, column=3, padx=12, pady=10, sticky="ew")
        self.progress.set(0)
        self.status_lbl = ctk.CTkLabel(actions, text="Idle", width=170, anchor="w")
        self.status_lbl.grid(row=0, column=4, padx=(0, 12))

        log_frame = ctk.CTkFrame(self)
        log_frame.grid(row=3, column=0, padx=14, pady=(6, 14), sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(log_frame, text="Log", anchor="w").grid(row=0, column=0, padx=12, pady=(8, 0), sticky="w")
        self.log_box = ctk.CTkTextbox(log_frame, wrap="word", state="disabled", font=("Consolas", 12))
        self.log_box.grid(row=1, column=0, padx=12, pady=(4, 12), sticky="nsew")

    # ---------------- UI state ----------------
    def _refresh_states(self):
        is_srt = self.input_path.get().lower().endswith(".srt")
        if is_srt:
            self.chk_audio.deselect()
            self.chk_subs.deselect()
        for w in (self.chk_audio, self.chk_subs):
            w.configure(state="disabled" if is_srt else "normal")
        self.audio_fmt_menu.configure(state="normal" if self.chk_audio.get() else "disabled")
        subs_on = bool(self.chk_subs.get()) and not is_srt
        trans_on = bool(self.chk_translate.get())
        tts_on = bool(self.chk_tts.get())
        asr_on = subs_on or ((trans_on or tts_on) and not is_srt)
        for w in (self.model_menu, self.device_menu):
            w.configure(state="normal" if asr_on else "disabled")
        self.source_menu.configure(state="normal" if (asr_on or trans_on or tts_on) else "disabled")
        trans_fields = trans_on or tts_on
        for w in (self.target_menu, self.engine_menu):
            w.configure(state="normal" if trans_fields else "disabled")
        self.tts_menu.configure(state="normal" if tts_on else "disabled")

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Choose a video, audio, or subtitle file", filetypes=FILETYPES
        )
        if path:
            self.input_path.set(path)
            if not self.output_dir.get():
                self.output_dir.set(str(Path(path).parent))
            self._refresh_states()
            self._log(f"Input: {path}")

    def _browse_output(self):
        path = filedialog.askdirectory(title="Choose an output folder")
        if path:
            self.output_dir.set(path)

    # ---------------- job control ----------------
    def _start(self):
        in_path = self.input_path.get()
        if not in_path or not Path(in_path).exists():
            messagebox.showwarning(APP_TITLE, "Please choose a valid input file.")
            return
        is_srt = in_path.lower().endswith(".srt")
        save_audio = bool(self.chk_audio.get()) and not is_srt
        transcribe = bool(self.chk_subs.get()) and not is_srt
        translate = bool(self.chk_translate.get())
        generate_audio = bool(self.chk_tts.get())
        if generate_audio and not translate:
            if not messagebox.askyesno(
                APP_TITLE,
                "'Translated audio' needs translated subtitles. Enable 'Translate subtitles' too?"
            ):
                return
            translate = True
            self.chk_translate.select()
        if is_srt and not (translate or generate_audio):
            messagebox.showwarning(APP_TITLE, "An .srt file is loaded - enable 'Translate subtitles' and/or 'Translated audio'.")
            return
        if not (save_audio or transcribe or translate or generate_audio):
            messagebox.showwarning(APP_TITLE, "Enable at least one task.")
            return
        out_dir = self.output_dir.get() or str(Path(in_path).parent)
        src_name = self.source_menu.get()
        cfg = JobConfig(
            input_path=in_path,
            output_dir=out_dir,
            save_audio=save_audio,
            audio_format=self.audio_fmt_menu.get(),
            transcribe=transcribe,
            translate=translate,
            source_language=None if src_name == "Auto-detect" else whisper_code(src_name),
            target_language=self.target_menu.get(),
            translation_engine=ENGINE_KEYS[self.engine_menu.get()],
            generate_audio=generate_audio,
            tts_engine=TTS_ENGINE_KEYS[self.tts_menu.get()],
            whisper_model=self.model_menu.get(),
            device=DEVICE_CHOICES[self.device_menu.get()],
        )
        self.cancel_event.clear()
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress.set(0)
        self._log("-" * 60)

        def work():
            try:
                outputs = run_job(cfg, log=self._qlog, progress=self._qprog, cancel=self.cancel_event)
                self.queue.put(("done", True, f"Finished. {len(outputs)} file(s) written."))
            except InterruptedError:
                self.queue.put(("done", False, "Cancelled by user."))
            except Exception as exc:
                self.queue.put(("done", False, f"Error: {exc}"))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _cancel(self):
        self.cancel_event.set()
        self.cancel_btn.configure(state="disabled")
        self._log("Cancelling...")

    # ---------------- queue / logging ----------------
    def _qlog(self, msg):
        self.queue.put(("log", msg))

    def _qprog(self, frac, text=""):
        self.queue.put(("progress", max(0.0, min(1.0, frac)), text))

    def _poll_queue(self):
        try:
            while True:
                item = self.queue.get_nowait()
                if item[0] == "log":
                    self._log(item[1])
                elif item[0] == "progress":
                    self.progress.set(item[1])
                    if item[2]:
                        self.status_lbl.configure(text=item[2])
                elif item[0] == "done":
                    self._on_done(item[1], item[2])
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{ts}] {msg}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _on_done(self, ok, message):
        self._log(message)
        self.status_lbl.configure(text="Done" if ok else "Stopped")
        if ok:
            self.progress.set(1)
        self.start_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        if ok:
            messagebox.showinfo(APP_TITLE, message)
        else:
            messagebox.showwarning(APP_TITLE, message)

    def _open_output(self):
        path = self.output_dir.get() or self.input_path.get()
        if not path:
            return
        p = Path(path)
        folder = p if p.is_dir() else p.parent
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(folder))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as exc:
            self._log(f"Could not open folder: {exc}")

    def _on_close(self):
        self.cancel_event.set()
        self.destroy()


def main():
    app = SubtitleStudioApp()
    app.mainloop()


if __name__ == "__main__":
    main()
