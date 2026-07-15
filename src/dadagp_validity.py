"""
DadaGP token-format validity checker.

The referee for the "legal move rate" of symbolic metal: a generated
token stream is scored on how much of it is well-formed DadaGP.

Two layers:
  1. token-level  : fraction of tokens that match the DadaGP grammar
  2. sequence-level: is the whole generation structurally plausible
                     (mostly-valid tokens + has a measure + has a note)

Calibration goal: real DadaGP records score ~1.0; English/garbage scores ~0.0.
Pure-Python, no dependencies — runs in any env. The gold-standard check
(does the official decoder accept it?) lives separately and needs PyGuitarPro.
"""
import re

# --- token grammar (calibrated against real DadaGP records) ------------------
# instrument prefix: letters + optional index, e.g. distorted0, clean1, bass, drums, leads0
_INSTR = r"[a-z]+\d*"

TOKEN_PATTERNS = [
    r"artist:.+",                                   # header: artist
    r"downtune:-?\d+",                              # header: global semitone downtune
    r"tempo:\d+",                                   # header: bpm
    r"start", r"end", r"new_measure",              # structural control
    r"measure:repeat_open",
    r"measure:repeat_close:\d+",
    r"measure:repeat_alternative:\d+",
    r"wait:\d+",                                    # time advance in ticks (960/quarter)
    rf"{_INSTR}:note:s[1-7]:f-?\d+",                # pitched note: string 1-7 (frets -1/-2 = drop tuning)
    r"drums:note:\d+",                              # drum hit: MIDI percussion number
    rf"{_INSTR}:rest",                              # instrument rest
    r"nfx:.+",                                      # note effects (palm_mute, dead, tie, slide, bend...)
    r"bfx:.+",                                      # beat effects (stroke, slap_effect...)
    r"param:.+",                                    # multi-token effect parameters (grace/bend)
]
_TOKEN_RE = re.compile(r"^(?:%s)$" % "|".join(TOKEN_PATTERNS))

# extra range sanity for pitched notes (a guitar/bass isn't 90 frets;
# -1/-2 encode drop tuning on the lowest string, nothing lower exists)
_NOTE_RE = re.compile(rf"^{_INSTR}:note:s([1-7]):f(-?\d+)$")
MAX_FRET = 36  # generous; real tabs rarely exceed ~24
MIN_FRET = -2


def is_valid_token(tok: str) -> bool:
    """True if `tok` is a well-formed DadaGP token (with light range sanity)."""
    tok = tok.strip()
    if not tok or not _TOKEN_RE.match(tok):
        return False
    m = _NOTE_RE.match(tok)
    if m and not (MIN_FRET <= int(m.group(2)) <= MAX_FRET):
        return False
    return True


def tokenize(text: str) -> list[str]:
    """Split a raw generation into candidate tokens (whitespace/newline separated)."""
    return [t for t in re.split(r"\s+", text.strip()) if t]


def validate_sequence(text: str) -> dict:
    """
    Score one generation. Returns metrics + a single `valid` boolean.

    valid == structurally plausible DadaGP: >=90% of tokens well-formed,
    and it actually contains musical content (a measure and a note).
    """
    toks = tokenize(text)
    n = len(toks)
    if n == 0:
        return dict(n_tokens=0, token_validity=0.0, has_measure=False,
                    has_note=False, valid=False)

    n_valid = sum(is_valid_token(t) for t in toks)
    token_validity = n_valid / n
    has_measure = any(t.strip() == "new_measure" for t in toks)
    has_note = any(":note:" in t for t in toks if is_valid_token(t))

    valid = token_validity >= 0.90 and has_measure and has_note
    return dict(n_tokens=n, token_validity=round(token_validity, 4),
                has_measure=has_measure, has_note=has_note, valid=valid)


def summarize(results: list[dict]) -> dict:
    """Aggregate per-generation results into headline baseline numbers."""
    k = len(results)
    if k == 0:
        return dict(n=0)
    return dict(
        n=k,
        mean_token_validity=round(sum(r["token_validity"] for r in results) / k, 4),
        seq_validity_rate=round(sum(r["valid"] for r in results) / k, 4),
        pct_with_measure=round(sum(r["has_measure"] for r in results) / k, 4),
        pct_with_note=round(sum(r["has_note"] for r in results) / k, 4),
    )


if __name__ == "__main__":
    # self-test / calibration: run against a real record and against garbage
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as f:
            print(sys.argv[1], "->", validate_sequence(f.read()))
    else:
        print("usage: python dadagp_validity.py <tokens.txt>")
