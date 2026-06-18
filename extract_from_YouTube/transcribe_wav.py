"""Transcribe the pre-extracted wav with faster-whisper (same settings as
acquire.from_whisper: base.en + VAD). Dumps timestamped segments to JSON."""
import json, time
from pathlib import Path
from faster_whisper import WhisperModel

WAV = "/tmp/ptcg_audio.wav"
OUT = "/tmp/ptcg_transcript.json"

t0 = time.time()
print("loading base.en model...", flush=True)
wm = WhisperModel("base.en", compute_type="int8")
print("transcribing...", flush=True)
segments, info = wm.transcribe(WAV, vad_filter=True)
out = []
for s in segments:
    out.append({"start": s.start, "end": s.end, "text": s.text.strip()})
    if len(out) % 50 == 0:
        print(f"  {len(out)} segs, t={s.end:.0f}s, elapsed={time.time()-t0:.0f}s", flush=True)
Path(OUT).write_text(json.dumps(out, indent=2))
print(f"done: {len(out)} segments in {time.time()-t0:.0f}s -> {OUT}", flush=True)
