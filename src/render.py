"""
Render DadaGP tokens to an audible WAV — the "press play" path.

Self-contained: maps string:fret -> pitch (standard tuning + downtune),
lays notes out in time (960 ticks/quarter at the header tempo), and
synthesizes with numpy (sawtooth + tanh soft-clip = distorted-guitar buzz;
noise bursts for drums). No soundfont / fluidsynth needed. Scrappy but real.

  python render.py <tokens.txt> <out.wav> [max_seconds]
"""
import sys
import numpy as np
from scipy.io import wavfile

SR = 44100
TPQ = 960  # ticks per quarter note

# standard tuning, string s1 (highest) .. s6/s7 (lowest), as MIDI note numbers
GUITAR = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40, 7: 35}
BASS   = {1: 43, 2: 38, 3: 33, 4: 28, 5: 23, 6: 18}


def parse(text: str):
    """-> (events, tempo). events: list of dicts {kind,pitch/drum,start,dur} in seconds."""
    toks = [t.strip() for t in text.split() if t.strip()]
    tempo, downtune = 120, 0
    for t in toks:
        if t.startswith("tempo:"):
            try: tempo = int(t.split(":")[1])
            except ValueError: pass
        elif t.startswith("downtune:"):
            try: downtune = int(t.split(":")[1])
            except ValueError: pass
    spt = (60.0 / max(tempo, 1)) / TPQ  # seconds per tick

    events, pending, cur = [], [], 0
    def flush(dur_ticks):
        d = max(dur_ticks, TPQ // 8) * spt
        for e in pending:
            e["dur"] = d
            events.append(e)
        pending.clear()

    for t in toks:
        if t.startswith("wait:"):
            try: n = int(t.split(":")[1])
            except ValueError: n = TPQ // 4
            flush(n); cur += n
        elif t.startswith("drums:note:"):
            try: dn = int(t.split(":")[3] if t.count(":") >= 3 else t.split(":")[-1])
            except ValueError: continue
            pending.append(dict(kind="drum", drum=dn, start=cur * spt, dur=0.12))
        elif ":note:s" in t:
            p = t.split(":")
            instr = p[0]
            try:
                string = int(p[2][1:]); fret = int(p[3][1:])
            except (ValueError, IndexError):
                continue
            table = BASS if instr.startswith("bass") else GUITAR
            pitch = table.get(string, 45) + fret - downtune
            kind = "bass" if instr.startswith("bass") else ("clean" if instr.startswith("clean") else "gtr")
            pending.append(dict(kind=kind, pitch=pitch, start=cur * spt, dur=0.25))
    flush(TPQ // 2)
    return events, tempo


def _env(n, attack=0.005, release=0.04):
    e = np.ones(n)
    a = min(int(SR * attack), n)
    r = min(int(SR * release), n)
    if a: e[:a] = np.linspace(0, 1, a)
    if r: e[-r:] *= np.linspace(1, 0, r)
    return e


def _tone(pitch, dur, drive, amp):
    n = max(int(SR * dur), 1)
    t = np.arange(n) / SR
    f = 440.0 * 2 ** ((pitch - 69) / 12.0)
    saw = 2.0 * (t * f - np.floor(0.5 + t * f))      # sawtooth [-1,1]
    return np.tanh(drive * saw) * _env(n) * amp       # soft-clip distortion


def _drum(dn, dur):
    n = max(int(SR * dur), 1)
    noise = np.random.uniform(-1, 1, n) * _env(n, 0.001, 0.05)
    if dn in (35, 36):                                # kick: low sine thump
        t = np.arange(n) / SR
        return (np.sin(2 * np.pi * 70 * t) * np.exp(-30 * t)) * 0.9
    if dn in (38, 40):                                # snare
        return noise * 0.6
    if dn in (42, 44, 46):                            # hi-hat: short bright
        return (noise * np.linspace(1, 0, n)) * 0.3
    return noise * 0.5                                # crash/other


def render(text: str, out_path: str, max_seconds: float = 30.0):
    events, tempo = parse(text)
    events = [e for e in events if e["start"] < max_seconds]
    if not events:
        raise ValueError("no renderable events")
    total = min(max(e["start"] + e["dur"] for e in events), max_seconds) + 0.2
    buf = np.zeros(int(SR * total))
    for e in events:
        i = int(e["start"] * SR)
        if e["kind"] == "drum":
            w = _drum(e["drum"], e["dur"])
        else:
            drive, amp = {"gtr": (6.0, 0.5), "clean": (2.0, 0.4), "bass": (3.0, 0.55)}[e["kind"]]
            w = _tone(e["pitch"], e["dur"], drive, amp)
        j = min(i + len(w), len(buf))
        buf[i:j] += w[: j - i]
    buf = buf / (np.max(np.abs(buf)) + 1e-9) * 0.9
    wavfile.write(out_path, SR, (buf * 32767).astype(np.int16))
    print(f"rendered {len(events)} events, {total:.1f}s @ {tempo}bpm -> {out_path}")


if __name__ == "__main__":
    txt = open(sys.argv[1], encoding="utf-8").read()
    out = sys.argv[2] if len(sys.argv) > 2 else "out.wav"
    render(txt, out, float(sys.argv[3]) if len(sys.argv) > 3 else 30.0)
