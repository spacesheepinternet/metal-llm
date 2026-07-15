"""
Prepare a DadaGP token corpus for training.

- scans an input dir for *.tokens.txt (one song per file)
- extracts the native header metadata (artist / downtune / tempo)
- optionally prepends a `genre:<slug>` conditioning line, looked up in a JSON
  mapping of filename-stem-or-artist -> genre (the corpus ships per-artist
  folders; a mapping file gets built when it lands)
- splits train/val BY SONG (windows from the same song never straddle the
  split) and writes one JSONL record per song

The `genre:` line is a control token for conditioning only — it is NOT part
of the DadaGP grammar. Strip it (strip_control_tokens) before handing text
to the official decoder.

Usage:
  python src/prepare_data.py --in data/samples --out data/prepared      # dry-run on the 2 samples
  python src/prepare_data.py --in data/corpus --genres data/genres.json
"""
import argparse, glob, hashlib, json, os, re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_FRAC = 0.05

# frets -1/-2 on the lowest string are how DadaGP encodes drop tunings
_DROP_RE = re.compile(r":note:s[1-7]:f-[12]\b")


def extract_meta(text: str) -> dict:
    meta = {"artist": None, "downtune": None, "tempo": None}
    for tok in text.split():
        if tok.startswith("artist:") and meta["artist"] is None:
            meta["artist"] = tok.split(":", 1)[1]
        elif tok.startswith("downtune:") and meta["downtune"] is None:
            meta["downtune"] = tok.split(":", 1)[1]
        elif tok.startswith("tempo:") and meta["tempo"] is None:
            meta["tempo"] = tok.split(":", 1)[1]
        if all(v is not None for v in meta.values()):
            break
    return meta


def lookup_genre(stem: str, meta: dict, genres: dict) -> str | None:
    for key in (stem, meta.get("artist")):
        if key and key in genres:
            return genres[key]
    return None


def strip_control_tokens(text: str) -> str:
    """Remove conditioning-only lines so the result is pure DadaGP again."""
    return "\n".join(l for l in text.splitlines()
                     if not l.strip().startswith("genre:"))


def split_of(stem: str) -> str:
    """Deterministic per-song split — stable across runs and machines."""
    h = int(hashlib.sha256(stem.encode()).hexdigest(), 16) % 10_000
    return "val" if h < VAL_FRAC * 10_000 else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default=os.path.join(REPO_ROOT, "data", "corpus"))
    ap.add_argument("--out", dest="outdir", default=os.path.join(REPO_ROOT, "data", "prepared"))
    ap.add_argument("--genres", default=None, help="JSON mapping stem-or-artist -> genre slug")
    args = ap.parse_args()

    genres = {}
    if args.genres:
        with open(args.genres, encoding="utf-8") as f:
            genres = json.load(f)

    paths = sorted(glob.glob(os.path.join(args.indir, "**", "*.tokens.txt"), recursive=True))
    if not paths:
        raise SystemExit(f"no *.tokens.txt under {args.indir}")

    os.makedirs(args.outdir, exist_ok=True)
    writers = {s: open(os.path.join(args.outdir, f"{s}.jsonl"), "w", encoding="utf-8")
               for s in ("train", "val")}
    counts = {"train": 0, "val": 0, "with_genre": 0}

    for p in paths:
        stem = os.path.basename(p).removesuffix(".tokens.txt")
        with open(p, encoding="utf-8") as f:
            text = f.read().strip()
        meta = extract_meta(text)
        meta["drop_tuning"] = bool(_DROP_RE.search(text))
        genre = lookup_genre(stem, meta, genres)
        if genre:
            text = f"genre:{genre}\n{text}"
            counts["with_genre"] += 1
        split = split_of(stem)
        rec = {"stem": stem, "meta": meta, "genre": genre,
               "n_ws_tokens": len(text.split()), "text": text}
        writers[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[split] += 1

    for w in writers.values():
        w.close()
    print(f"[prepare] {counts['train']} train / {counts['val']} val songs "
          f"({counts['with_genre']} with genre tag) -> {args.outdir}")


if __name__ == "__main__":
    main()
