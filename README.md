# Chess-LLM — teaching a small language model to play chess

Fine-tune a small open LLM to play legal, competent chess — and **measure it rigorously**
against the Stockfish engine. A base LLM constantly plays *illegal* moves; the goal here is
to teach the rules and reasonable amateur play, and to prove it with hard numbers
(legal-move rate, win-rate vs Stockfish, estimated Elo).

> **Status: 🚧 work in progress.** Targets below are goals, not yet results.

## Why
A portfolio project focused on the modeling skill that application-layer projects don't show:
**fine-tuning + rigorous evaluation.** Chess is fun, data-rich, and — crucially — *objectively
measurable* (a move is legal or not; a game is won or not), which is what makes the results
credible.

## Approach
- **Base model:** a small open LLM (Qwen2.5-3B/7B or Llama-3.x-8B).
- **Data:** the [Lichess open database](https://database.lichess.org/) — millions of real games
  (PGN), filterable by player rating.
- **Method:** QLoRA fine-tuning on next-move prediction (single GPU).
- **Evaluation:** [`python-chess`](https://python-chess.readthedocs.io/) for legality + game play,
  and [Stockfish](https://stockfishchess.org/) as opponent and skill yardstick.

## Metrics (goals)
| Metric | Base model | Fine-tuned (goal) |
|---|---|---|
| Legal-move rate | ~85% | **99%+** |
| Win-rate vs random-move bot | — | **high** |
| Win/draw vs Stockfish (low levels) | ~0 | **measurable** |
| Estimated Elo vs Stockfish | — | **~1200+** |

## Project structure
```
chess-llm/
├── data/     # Lichess PGN dumps + processed training data (gitignored)
├── src/      # data prep, training (QLoRA), and eval harness
├── README.md
├── requirements.txt
└── HANDOFF.md  # full brief + step-by-step build guide
```

## Setup
```bash
python -m venv venv
# Windows:  venv\Scripts\activate      Linux/Mac:  source venv/bin/activate
pip install -r requirements.txt
# Also install the Stockfish engine binary (https://stockfishchess.org/download/)
```
> Note: `bitsandbytes` (QLoRA) is easiest on Linux + NVIDIA GPU — plan to train on
> Colab / Modal / RunPod or a local NVIDIA card.

## Roadmap
1. Env + download a (small) Lichess PGN dump.
2. **Baseline:** measure the base model's legal-move rate — the "before" number.
3. QLoRA fine-tune on next-move prediction.
4. Eval harness: legal-move rate + game loop vs random bot / Stockfish.
5. Writeup with before/after charts (+ a small playable demo).

See `HANDOFF.md` for the detailed build guide.

## License
MIT (see below) — code is free to use; Lichess data has its own terms.
