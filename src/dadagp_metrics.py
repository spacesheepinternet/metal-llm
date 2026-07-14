"""
Deeper eval metrics — the ones that actually separate a base model from a
fine-tuned one (validity/decodability already saturate under one-shot).

  meter_report    : do the wait-ticks in each bar sum to a valid bar length?
  richness_report : is it a real multi-track song, or a 15-token toy?

Calibrated on the real progmetal record: 56/56 measures == 3840 ticks (4/4),
5 instruments, 3904 tokens.
"""
import collections

TICKS_PER_QUARTER = 960
# common bar lengths in ticks (4/4, 3/4/6/8, 2/4, 7/8, 5/4, 12/8, 2/2-double)
STANDARD_BARS = {1920, 2880, 3360, 3840, 4800, 5760, 7680}


def _measure_tick_sums(tokens: list[str]) -> list[int]:
    sums, cur, started = [], 0, False
    for t in tokens:
        if t == "new_measure":
            if started:
                sums.append(cur)
            cur, started = 0, True
        elif t.startswith("wait:"):
            try:
                cur += int(t.split(":")[1])
            except ValueError:
                pass
    if started:
        sums.append(cur)
    return sums


def meter_report(tokens: list[str]) -> dict:
    sums = _measure_tick_sums(tokens)
    if not sums:
        return dict(n_measures=0, pct_standard_bars=0.0, meter_consistency=0.0)
    modal = collections.Counter(sums).most_common(1)[0][0]
    return dict(
        n_measures=len(sums),
        pct_standard_bars=round(sum(s in STANDARD_BARS for s in sums) / len(sums), 4),
        meter_consistency=round(sum(s == modal for s in sums) / len(sums), 4),  # self-consistency
        modal_bar_ticks=modal,
    )


def richness_report(tokens: list[str]) -> dict:
    instruments = set()
    n_notes = 0
    for t in tokens:
        if ":note:" in t:
            instruments.add(t.split(":")[0])
            n_notes += 1
        elif t.endswith(":rest"):
            instruments.add(t.split(":")[0])
    n_measures = sum(1 for t in tokens if t == "new_measure")
    has_gtr = any(i.startswith(("distorted", "clean", "guitar")) for i in instruments)
    return dict(
        n_tokens=len(tokens),
        n_measures=n_measures,
        n_instruments=len(instruments),
        instruments=sorted(instruments),
        n_notes=n_notes,
        multitrack=has_gtr and "bass" in instruments and "drums" in instruments,
    )


def full_report(text_or_tokens) -> dict:
    toks = text_or_tokens if isinstance(text_or_tokens, list) else text_or_tokens.split()
    return {"meter": meter_report(toks), "richness": richness_report(toks)}


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1], encoding="utf-8") as f:
        print(json.dumps(full_report(f.read()), indent=2))
