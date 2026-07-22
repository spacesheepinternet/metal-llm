"""
Prepare the DadaGP v1.1 corpus for training.

Pipeline (order matters):
  1. ingest the official `_DadaGP_all_metadata.json` (relpath -> artist_token,
     genre_tokens LIST, validation_set flag)
  2. dedup: cluster files by (artist folder, normalized title) — ~13% of the
     corpus is duplicate versions — and keep the largest file per cluster
  3. split train/val BY CLUSTER (deterministic hash), because the official
     split leaks: 578 duplicate clusters straddle its train/val boundary
  4. drop songs under --min-tokens (default 200: fragments and exercises)
  5. prepend curated `genre:` control lines (see curate_genres) and write one
     JSONL record per song

Genre policy ("curated tag-set + unknown kept"):
  - keep metal-family tags, *_rock / punk / grunge tags, and unknown_genre
  - drop Spotify-oddity co-tags (permanent_wave, mellow_gold, sleep, ...)
  - cap at 3 tags per song, most specific (rarest) first
  - songs whose tags all get dropped fall back to genre:unknown_genre, so
    every record has at least one genre line and the model always sees the
    conditioning slot

Control lines are NOT DadaGP grammar — strip_control_tokens() before handing
text to the official decoder.

Usage:
  python src/prepare_data.py                       # full corpus
  python src/prepare_data.py --in data/samples     # dry-run on the 2 samples
"""
import argparse, hashlib, json, os, re
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(REPO_ROOT, "data", "corpus", "DadaGP-v1.1")
VAL_FRAC = 0.05

# frets -1/-2 on the lowest string are how DadaGP encodes drop tunings
_DROP_RE = re.compile(r":note:s[1-7]:f-[12]\b")
# "(2)", "(live)", "[acoustic]" style version markers in titles
_PAREN_RE = re.compile(r"[\(\[].*?[\)\]]")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# metal-family tags that don't contain the substring "metal"
_EXTRA_METAL = {"old_school_thrash", "grindcore", "deathcore", "djent"}
# frequent Spotify co-tags that carry no useful style signal for us
_TAG_BLOCKLIST = {"permanent_wave", "mellow_gold", "neo_mellow", "sleep",
                  "adult_standards", "lilith", "new_romantic"}


def norm_title(relpath: str) -> tuple[str, str]:
    """(artist_folder, normalized_title) — the dedup cluster key."""
    parts = relpath.replace("\\", "/").split("/")
    artist_dir = parts[1] if len(parts) >= 3 else "unknown"
    base = re.sub(r"\.gp[x345]?\.tokens\.txt$", "", parts[-1], flags=re.I)
    title = base.split(" - ", 1)[1] if " - " in base else base
    title = _PAREN_RE.sub("", title.lower())
    title = _NON_ALNUM_RE.sub("", title)
    return (_NON_ALNUM_RE.sub("", artist_dir.lower()), title)


def keep_tag(slug: str) -> bool:
    if slug in _TAG_BLOCKLIST:
        return False
    return ("metal" in slug or slug in _EXTRA_METAL
            or slug == "rock" or slug.endswith("_rock")
            or "punk" in slug or "grunge" in slug
            or slug == "unknown_genre")


def curate_genres(genre_tokens: list[str], tag_freq: Counter) -> list[str]:
    """Curated tags, most specific (rarest) first, capped at 3."""
    kept = [g.split(":", 1)[1] for g in genre_tokens]
    kept = [s for s in kept if keep_tag(s)]
    kept.sort(key=lambda s: (tag_freq.get(s, 0), s))
    return kept[:3] or ["unknown_genre"]


def strip_control_tokens(text: str) -> str:
    """Remove conditioning-only lines so the result is pure DadaGP again."""
    return "\n".join(l for l in text.splitlines()
                     if not l.strip().startswith("genre:"))


def split_of(cluster_key: tuple[str, str]) -> str:
    """Deterministic per-CLUSTER split — versions of a song never straddle it."""
    h = int(hashlib.sha256("/".join(cluster_key).encode()).hexdigest(), 16) % 10_000
    return "val" if h < VAL_FRAC * 10_000 else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default=CORPUS_DIR)
    ap.add_argument("--out", dest="outdir", default=os.path.join(REPO_ROOT, "data", "prepared"))
    ap.add_argument("--min-tokens", type=int, default=200)
    args = ap.parse_args()

    meta_path = os.path.join(args.indir, "_DadaGP_all_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            all_meta = json.load(f)
    else:  # sample-dir dry runs have no metadata file
        all_meta = {}
        for root, _, files in os.walk(args.indir):
            for fn in files:
                if fn.endswith(".tokens.txt"):
                    rel = os.path.relpath(os.path.join(root, fn), args.indir)
                    all_meta[rel.replace("\\", "/")] = {"genre_tokens": []}
    if not all_meta:
        raise SystemExit(f"no songs under {args.indir}")

    # global tag frequency (for rarest-first ordering)
    tag_freq = Counter(g.split(":", 1)[1]
                       for m in all_meta.values() for g in m.get("genre_tokens", []))

    # pass 1 — dedup by cluster, keep the largest file (size ~ token count)
    clusters: dict[tuple, tuple[int, str]] = {}
    missing = 0
    for rel in all_meta:
        p = os.path.join(args.indir, rel)
        if not os.path.exists(p):
            missing += 1
            continue
        key = norm_title(rel)
        size = os.path.getsize(p)
        if key not in clusters or size > clusters[key][0]:
            clusters[key] = (size, rel)

    # pass 2 — read survivors, filter, write
    os.makedirs(args.outdir, exist_ok=True)
    writers = {s: open(os.path.join(args.outdir, f"{s}.jsonl"), "w", encoding="utf-8")
               for s in ("train", "val")}
    counts = Counter()
    tag_written = Counter()
    for key, (_, rel) in sorted(clusters.items()):
        with open(os.path.join(args.indir, rel), encoding="utf-8") as f:
            text = f.read().strip()
        n_tok = len(text.split())
        if n_tok < args.min_tokens:
            counts["dropped_short"] += 1
            continue
        m = all_meta[rel]
        genres = curate_genres(m.get("genre_tokens", []), tag_freq)
        tag_written.update(genres)
        text = "\n".join(f"genre:{g}" for g in genres) + "\n" + text
        split = split_of(key)
        rec = {"rel": rel,
               "artist": (m.get("artist_token") or "artist:unknown_artist").split(":", 1)[1],
               "genres": genres,
               "drop_tuning": bool(_DROP_RE.search(text)),
               "official_val": bool(m.get("validation_set")),
               "n_ws_tokens": n_tok,
               "text": text}
        writers[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[split] += 1
    for w in writers.values():
        w.close()

    report = {
        "input_songs": len(all_meta),
        "missing_files": missing,
        "duplicates_removed": len(all_meta) - missing - len(clusters),
        "dropped_short": counts["dropped_short"],
        "train": counts["train"], "val": counts["val"],
        "top_tags_written": dict(tag_written.most_common(25)),
    }
    with open(os.path.join(args.outdir, "prep_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
