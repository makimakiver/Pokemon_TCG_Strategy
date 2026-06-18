"""Acquire a timestamped transcript from a YouTube match video.

Two paths:
  1. YouTube auto/uploaded subtitles via yt-dlp  (fast, no model, default)
  2. Whisper ASR on the downloaded audio          (slower, better, optional)

Both return: List[{"start": float, "end": float, "text": str}]
"""
from __future__ import annotations
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _parse_vtt(path: Path) -> List[Dict]:
    """Minimal WebVTT parser -> timestamped segments."""
    ts = re.compile(r"(\d+):(\d+):(\d+)[.,](\d+)\s*-->\s*(\d+):(\d+):(\d+)[.,](\d+)")

    def secs(h, m, s, ms):
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    segs, cur = [], None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = ts.search(line)
        if m:
            if cur and cur["text"].strip():
                segs.append(cur)
            cur = {"start": secs(*m.groups()[:4]), "end": secs(*m.groups()[4:]), "text": ""}
        elif cur is not None and line.strip() and "-->" not in line:
            clean = re.sub(r"<[^>]+>", "", line).strip()  # strip karaoke tags
            if clean and clean not in cur["text"]:
                cur["text"] += (" " + clean)
    if cur and cur["text"].strip():
        segs.append(cur)
    # collapse exact-duplicate rolling-caption lines
    out: List[Dict] = []
    for s in segs:
        s["text"] = s["text"].strip()
        if not out or out[-1]["text"] != s["text"]:
            out.append(s)
    return out


def from_subtitles(url: str, lang: str = "en") -> List[Dict]:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "subs"
        _run(["yt-dlp", "--skip-download", "--write-auto-subs", "--write-subs",
              "--sub-langs", lang, "--sub-format", "vtt",
              "-o", str(out), url])
        vtts = list(Path(d).glob("*.vtt"))
        if not vtts:
            raise RuntimeError("no subtitles found; try asr='whisper'")
        return _parse_vtt(vtts[0])


def from_whisper(url: str, model: str = "base.en") -> List[Dict]:
    """Requires `faster-whisper`. Downloads audio, transcribes with timestamps."""
    from faster_whisper import WhisperModel  # lazy import
    with tempfile.TemporaryDirectory() as d:
        audio = Path(d) / "audio.m4a"
        _run(["yt-dlp", "-f", "bestaudio", "-x", "--audio-format", "m4a",
              "-o", str(audio), url])
        wm = WhisperModel(model)
        segments, _ = wm.transcribe(str(audio), vad_filter=True)
        return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]


def get_transcript(url: str, asr: str = "subs", lang: str = "en") -> List[Dict]:
    if asr == "whisper":
        return from_whisper(url)
    return from_subtitles(url, lang=lang)
