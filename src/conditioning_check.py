"""
Conditioning-accuracy checker: did the generation follow the requested knobs?

Reads a kill-test results JSON (each generation records the tempo/downtune it
was asked for) and scores whether the output honored the request. Today the
model can satisfy tempo/downtune by echoing the header — weak, but the base
model sometimes fails even that, so it's tracked. Genre and drop-tuning
checks slot in once conditioned training runs exist.

  python src/conditioning_check.py results/after_smoke_adapter.json
"""
import json, re, sys

_DROP_RE = re.compile(r":note:s[1-7]:f-[12]\b")


def first_header(text: str, key: str) -> str | None:
    for tok in text.split():
        if tok.startswith(f"{key}:"):
            return tok.split(":", 1)[1]
    return None


def check(gen: dict) -> dict:
    text = gen["text"]
    return {
        "tempo_ok": first_header(text, "tempo") == str(gen["tempo"]),
        "downtune_ok": first_header(text, "downtune") == str(gen["downtune"]),
        "uses_drop_tuning": bool(_DROP_RE.search(text)),
    }


def main(path: str):
    data = json.load(open(path, encoding="utf-8"))
    print(f"===== CONDITIONING CHECK — {data['model']} "
          f"adapter={data.get('adapter')} =====")
    by_cond = {}
    for g in data["generations"]:
        by_cond.setdefault(g["cond"], []).append(check(g))
    for cond, rs in by_cond.items():
        n = len(rs)
        print(f"{cond:10s}: tempo echo {sum(r['tempo_ok'] for r in rs)}/{n}  "
              f"downtune echo {sum(r['downtune_ok'] for r in rs)}/{n}  "
              f"drop-tuning used {sum(r['uses_drop_tuning'] for r in rs)}/{n}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/baseline_qwen3b.json")
