"""
Decodability gold-check: the acid test the token-validity metric can't do.

A generation is "valid-looking" if its tokens match the grammar; it is
GENUINELY valid only if the official DadaGP decoder turns it into a playable
GuitarPro file. This runs the real decoder on the base model's outputs.

Minimal normalization: DadaGP requires an `artist:` token first; our prompt
didn't ask for one, so we prepend `artist:unknown` if missing (a trivial
completeness fix) and otherwise leave the model's output untouched — so we're
measuring musical/structural decodability, not a missing header line.

Requires the official DadaGP encoder/decoder (MIT):
  git clone https://github.com/dada-bots/dadaGP external/dadaGP
and a Python env with PyGuitarPro.

Usage:
  python src/decodability_check.py --results results/baseline_qwen3b.json
"""
import argparse, json, os, subprocess, sys, tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def normalize(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    if not lines[0].startswith("artist:"):
        lines = ["artist:unknown"] + lines
    return "\n".join(lines)


def try_decode(tokens: str, dadagp_dir: str) -> tuple[bool, str]:
    """Return (success, error_snippet) for one generation."""
    dadagp = os.path.join(dadagp_dir, "dadagp.py")
    norm = normalize(tokens)
    if not norm:
        return False, "empty"
    with tempfile.TemporaryDirectory() as d:
        tin = os.path.join(d, "gen.tokens.txt")
        tout = os.path.join(d, "gen.gp5")
        with open(tin, "w", encoding="utf-8") as f:
            f.write(norm)
        try:
            p = subprocess.run([sys.executable, dadagp, "decode", tin, tout],
                               capture_output=True, text=True, timeout=120,
                               cwd=dadagp_dir)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        ok = p.returncode == 0 and os.path.exists(tout) and os.path.getsize(tout) > 0
        if ok:
            return True, ""
        err = (p.stderr or p.stdout).strip().splitlines()
        return False, (err[-1] if err else f"returncode={p.returncode}")[:160]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(REPO_ROOT, "results", "baseline_qwen3b.json"))
    ap.add_argument("--dadagp-dir", default=os.path.join(REPO_ROOT, "external", "dadaGP"))
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.dadagp_dir, "dadagp.py")):
        sys.exit(f"DadaGP decoder not found at {args.dadagp_dir} — "
                 "clone https://github.com/dada-bots/dadaGP there first.")

    data = json.load(open(args.results, encoding="utf-8"))
    by_cond = {}
    for g in data["generations"]:
        cond = g["cond"]
        ok, err = try_decode(g["text"], args.dadagp_dir)
        rec = by_cond.setdefault(cond, {"ok": 0, "n": 0, "errs": []})
        rec["n"] += 1
        rec["ok"] += ok
        if not ok:
            rec["errs"].append(err)

    print("===== DECODABILITY GOLD-CHECK (does it actually decode to a playable file?) =====")
    for cond, r in by_cond.items():
        rate = r["ok"] / r["n"] if r["n"] else 0
        print(f"\n{cond}: {r['ok']}/{r['n']} decode successfully ({rate:.0%})")
        if r["errs"]:
            print("  sample failures:")
            for e in r["errs"][:5]:
                print(f"    - {e}")


if __name__ == "__main__":
    main()
