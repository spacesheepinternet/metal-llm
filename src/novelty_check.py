"""
Novelty check: is the model composing or reciting?

Scores each generated song's token n-grams against the training songs the
model was actually trained on (the same seeded sample train_qlora.py used),
reporting the share of novel n-grams and the longest verbatim run copied
from any training song. Held-out val songs are scored the same way to give
the honest reference point: real, unseen music also shares idiomatic
n-grams (power-chord shapes, wait-patterns), so "novel" for a generation
should be judged against that background, not against 100%.

  python src/novelty_check.py --results results/pilot500_raw512_rp12_qwen3b.json
"""
import argparse, json, os, random

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEADER_PREFIXES = ("genre:", "artist:", "downtune:", "tempo:")


def music_tokens(text: str) -> list[str]:
    """Whitespace DadaGP tokens minus header/control lines."""
    return [t for t in text.split()
            if not t.startswith(HEADER_PREFIXES) and t not in ("start", "end")]


def ngrams(tokens: list[str], n: int):
    for i in range(len(tokens) - n + 1):
        yield hash(tuple(tokens[i:i + n]))


def sample_train_texts(path: str, n_songs: int, seed: int) -> list[str]:
    """Reproduce train_qlora.py's pilot sample (same two-pass seeded draw)."""
    with open(path, encoding="utf-8") as f:
        total = sum(1 for line in f if line.strip())
    take = set(range(total)) if not n_songs or n_songs >= total else \
        set(random.Random(seed).sample(range(total), n_songs))
    texts, i = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if i in take:
                texts.append(json.loads(line)["text"])
            i += 1
    return texts


def score(tokens: list[str], train_ngrams: set, n: int) -> dict:
    grams = list(ngrams(tokens, n))
    if not grams:
        return {"n_tokens": len(tokens), "n_ngrams": 0,
                "novel_rate": None, "longest_match": 0}
    hits = [g in train_ngrams for g in grams]
    longest = run = 0
    for h in hits:
        run = run + 1 if h else 0
        longest = max(longest, run)
    # longest run of R matched n-grams == verbatim copy of R+n-1 tokens
    return {"n_tokens": len(tokens), "n_ngrams": len(grams),
            "novel_rate": round(1 - sum(hits) / len(grams), 4),
            "longest_match": (longest + n - 1) if longest else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="kill-test results JSON with generations to score")
    ap.add_argument("--train-jsonl",
                    default=os.path.join(REPO_ROOT, "data", "prepared", "train.jsonl"))
    ap.add_argument("--val-jsonl",
                    default=os.path.join(REPO_ROOT, "data", "prepared", "val.jsonl"))
    ap.add_argument("--pilot-songs", type=int, default=500,
                    help="0 = index the full training set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ngram", type=int, default=8)
    ap.add_argument("--val-refs", type=int, default=12,
                    help="held-out songs scored as the background-overlap reference")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"[index] building {args.ngram}-gram index over "
          f"{args.pilot_songs or 'ALL'} training songs (seed {args.seed})")
    train_ngrams: set = set()
    for t in sample_train_texts(args.train_jsonl, args.pilot_songs, args.seed):
        train_ngrams.update(ngrams(music_tokens(t), args.ngram))
    print(f"[index] {len(train_ngrams):,} unique n-grams")

    data = json.load(open(args.results, encoding="utf-8"))
    gen_scores = []
    for g in data["generations"]:
        s = score(music_tokens(g["text"]), train_ngrams, args.ngram)
        gen_scores.append(s)

    val_scores = []
    for t in sample_train_texts(args.val_jsonl, args.val_refs, args.seed):
        # score the same-length slice a generation would produce
        val_scores.append(score(music_tokens(t)[:600], train_ngrams, args.ngram))

    def summarize(rows, label):
        rows = [r for r in rows if r["novel_rate"] is not None]
        if not rows:
            print(f"{label}: no scorable songs")
            return {}
        nov = sorted(r["novel_rate"] for r in rows)
        lng = max(r["longest_match"] for r in rows)
        s = {"n": len(rows), "median_novel_rate": nov[len(nov) // 2],
             "min_novel_rate": nov[0], "max_longest_match": lng}
        print(f"{label}: median novel-rate {s['median_novel_rate']:.2%}  "
              f"worst (least novel) {s['min_novel_rate']:.2%}  "
              f"longest verbatim copy {lng} tokens")
        return s

    print(f"\n===== NOVELTY ({args.ngram}-gram vs trained-on songs) =====")
    gsum = summarize(gen_scores, "generated ")
    vsum = summarize(val_scores, "held-out  ")
    print("(held-out row = real unseen songs; a generation is plagiarism-free "
          "when its novelty is in that neighborhood, not at 100%)")

    out = args.out or args.results.replace(".json", "_novelty.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"results": args.results, "ngram": args.ngram,
                   "pilot_songs": args.pilot_songs, "seed": args.seed,
                   "generated": {"summary": gsum, "songs": gen_scores},
                   "held_out": {"summary": vsum, "songs": val_scores}},
                  f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
