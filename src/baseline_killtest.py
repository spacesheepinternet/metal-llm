"""
Baseline kill-test: how badly does a *base* LLM fail at DadaGP token format?

This produces the "before" number for the fine-tune story:
  base model validity rate  --(fine-tune)-->  target 90%+

It prompts an instruct model to emit DadaGP tokens for a short metal passage
(zero-shot and one-shot), then scores each generation with the validity
referee. Every un-fine-tuned model is expected to score low here — that gap
is the whole point of the project.

Usage:
  python baseline_killtest.py --model Qwen/Qwen2.5-3B-Instruct --n 12
  python baseline_killtest.py --model Qwen/Qwen2.5-1.5B-Instruct --n 12   # smaller fallback
"""
import argparse, json, os, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dadagp_validity import validate_sequence, summarize

# a real snippet from the progmetal record, for the one-shot condition
ONE_SHOT_EXAMPLE = """downtune:0
tempo:100
start
new_measure
distorted0:note:s4:f2
distorted0:note:s5:f2
distorted0:note:s6:f0
bass:note:s5:f0
drums:note:36
wait:480
distorted0:note:s6:f3
wait:480
end"""

FORMAT_BLURB = (
    "DadaGP is a text format for guitar music. Tokens (one per line): "
    "`downtune:N` and `tempo:N` (header), `start`, `new_measure`, "
    "`<instrument>:note:sX:fY` (string X 1-6, fret Y) e.g. `distorted0:note:s6:f0`, "
    "`nfx:palm_mute` (effect on previous note), `wait:N` (advance time, 480=eighth), "
    "`drums:note:36` (kick), and `end`."
)


def build_messages(tempo: int, downtune: int, one_shot: bool):
    sys = ("You are a symbolic music generator. Output ONLY DadaGP tokens, "
           "one per line, and nothing else — no prose, no explanations, no markdown.")
    ask = (f"{FORMAT_BLURB}\n\nGenerate a short heavy-metal guitar passage "
           f"(~50 tokens) in DadaGP format with header downtune:{downtune} and "
           f"tempo:{tempo}. Start with the header, then `start`, the music, then `end`.")
    if one_shot:
        ask = f"Example of the format:\n{ONE_SHOT_EXAMPLE}\n\n{ask}"
    return [{"role": "system", "content": sys}, {"role": "user", "content": ask}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default=None,
                    help="path to a trained LoRA adapter — turns this into the 'after' run")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--raw-prefix", action="store_true",
                    help="prompt with a training-format header prefix instead of "
                         "chat messages — the native interface of a raw-LM "
                         "fine-tune; scores prefix+continuation as one song")
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="counter rest/wait degeneration loops (1.0 = off)")
    ap.add_argument("--out", default="baseline_results.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[env] device={device} dtype={dtype} model={args.model} adapter={args.adapter}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    if args.adapter:
        from peft import PeftModel
        if os.path.exists(os.path.join(args.adapter, "tokenizer_config.json")):
            # extend-vocab adapters ship their tokenizer; the embedding matrix
            # must be resized before PEFT overlays the trained token rows
            tok = AutoTokenizer.from_pretrained(args.adapter)
            model.resize_token_embeddings(len(tok))
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # vary the requested header per generation (also seeds the conditioning idea)
    headers = [(t, d) for t in (120, 160, 180, 200) for d in (0, 2)]
    results = {"zero_shot": [], "one_shot": []}
    generations = []

    conds = ("raw_prefix",) if args.raw_prefix else ("zero_shot", "one_shot")
    if args.raw_prefix:
        results = {"raw_prefix": []}
    for cond in conds:
        for i in range(args.n):
            tempo, downtune = headers[i % len(headers)]
            if cond == "raw_prefix":
                prompt = (f"genre:metal\nartist:unknown_artist\n"
                          f"downtune:{downtune}\ntempo:{tempo}\nstart\nnew_measure\n")
            else:
                msgs = build_messages(tempo, downtune, one_shot=(cond == "one_shot"))
                prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=True, temperature=0.8, top_p=0.95,
                                     repetition_penalty=args.repetition_penalty,
                                     pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            # strip common markdown fences the model may wrap around output
            gen = re.sub(r"```[a-z]*", "", gen)
            if cond == "raw_prefix":
                gen = prompt + gen  # the song is prefix + continuation
            r = validate_sequence(gen)
            results[cond].append(r)
            generations.append({"cond": cond, "tempo": tempo, "downtune": downtune,
                                 "metrics": r, "text": gen})
            print(f"[{cond}] gen {i+1}/{args.n}: token_validity={r['token_validity']:.2f} valid={r['valid']}")

    summary = {cond: summarize(rs) for cond, rs in results.items()}
    print("\n===== BASELINE ('before' number) =====")
    for cond, s in summary.items():
        print(f"{cond:10s}: mean_token_validity={s['mean_token_validity']}  "
              f"seq_validity_rate={s['seq_validity_rate']}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "adapter": args.adapter,
                   "summary": summary, "generations": generations},
                  f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
