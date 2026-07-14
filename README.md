# Metal-LLM — teaching a small language model to write metal

Fine-tune a small open LLM to generate **heavy-metal guitar music** in symbolic form — and
**measure it rigorously**. The model writes [DadaGP](https://github.com/dada-bots/dadaGP)
tokens (the text encoding behind 26k GuitarPro songs); every output is machine-checkable:
does it decode to a playable GuitarPro file, do the bars add up, is it a real multi-track
song or a 15-token toy?

> **Status: 🚧 baseline done, awaiting training corpus.** The eval harness is built and
> calibrated against real data; the base-model "before" numbers are in `results/`.

## Why
A portfolio project focused on the modeling skill that application-layer projects don't show:
**fine-tuning + rigorous evaluation.** Symbolic music has the same property that makes chess
or code credible eval domains — a formal grammar with a mechanical referee — but the output
is something you can actually *listen to*.

## Approach
- **Base model:** Qwen2.5-3B-Instruct (QLoRA on a single RTX 4070 Laptop GPU, 8 GB).
- **Format:** DadaGP tokens — `distorted0:note:s6:f0`, `wait:480` (960 ticks/quarter),
  `nfx:palm_mute`, `drums:note:36`, `new_measure`, headers `downtune:N` / `tempo:N`.
- **Data:** the DadaGP corpus (26k GuitarPro songs as token text; access is email-gated —
  request in progress). The encoder/decoder itself is open (MIT).
- **Referee:** the official DadaGP decoder (does it produce a playable `.gp5`?) plus custom
  metrics below.

## Eval design — what the baseline taught us
The obvious metric (token validity) **saturates**: the base model with a single in-context
example already emits ~90% decodable DadaGP. So validity is the sanity check, not the
headline. The metrics that actually separate base from fine-tuned:

| Metric | Base (Qwen2.5-3B) | Real songs | Fine-tuned (goal) |
|---|---|---|---|
| Zero-shot decodable-validity | **0%** | 100% | high, no example needed |
| Meter correctness (% bars with valid tick-sum) | **0%** | 100% | ~100% |
| Structural richness (tokens / song) | ~16 | ~3,900 | full multi-track songs |
| Conditioning accuracy (tempo / downtune / genre) | header-echo only | — | measurable control |
| Novelty (n-gram overlap vs training set) | — | — | low copying |

Baseline details: `results/baseline_qwen3b.json` (zero-shot vs one-shot, 10 generations each).

## Project structure
```
├── data/samples/          # two real DadaGP token files (from the MIT encoder repo)
├── external/dadaGP/       # official encoder/decoder — clone locally (gitignored)
├── results/               # baseline + (later) fine-tuned eval results
└── src/
    ├── dadagp_validity.py     # token/sequence grammar referee (pure Python)
    ├── dadagp_metrics.py      # meter correctness + structural richness
    ├── decodability_check.py  # gold check: run the official decoder on generations
    ├── baseline_killtest.py   # generate with the base model, score it (the "before")
    ├── train_qlora.py         # QLoRA training loop (4-bit NF4 + LoRA, 8GB-friendly)
    └── render.py              # tokens → WAV via a scrappy numpy synth (dev-time demo)
```

## Setup
```bash
python -m venv venv && venv\Scripts\activate   # or source venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/dada-bots/dadaGP external/dadaGP
```

Reproduce the baseline:
```bash
python src/baseline_killtest.py --model Qwen/Qwen2.5-3B-Instruct --n 10 --out results/baseline_qwen3b.json
python src/decodability_check.py --results results/baseline_qwen3b.json
python src/dadagp_metrics.py data/samples/progmetal.tokens.txt   # what "real" looks like
python src/render.py data/samples/progmetal.tokens.txt demo.wav  # press play
```

## Roadmap
1. ~~Eval harness calibrated on real DadaGP data~~ ✅
2. ~~Baseline kill-test on the base model (the "before" numbers)~~ ✅
3. ~~Training smoke-test: tiny QLoRA overfit on the sample songs~~ ✅
   (loss 0.57→0.15 in 40 steps, peak 4.4 GiB VRAM — 4-bit bitsandbytes works on Windows)
4. Corpus lands → real QLoRA fine-tune with conditioning tags (tempo, downtune, genre).
5. Re-run eval → "after" numbers; add novelty metric (n-gram overlap vs training set).
6. Writeup with before/after charts + audio demos (MIDI + guitar soundfont).

## License
MIT for the code here. DadaGP encoder/decoder is MIT (dada-bots); the DadaGP corpus is
research-gated with its own terms; sample token files are from the encoder repo's examples.
