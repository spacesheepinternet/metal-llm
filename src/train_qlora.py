"""
QLoRA fine-tune of a small instruct model on DadaGP token text.

Two modes:
  --smoke-test : tiny overfit run on the two sample songs (~40 steps).
                 Proves the full 4-bit + LoRA training loop works locally
                 (Windows + bitsandbytes + 8GB VRAM) before the corpus lands.
                 Pass criterion: training loss drops hard (memorization).
  (default)    : real training over data/corpus/ — wired up once the
                 DadaGP corpus arrives.

Optional:
  --extend-vocab : tokenizer-extension experiment ("option B"). Adds all
                 3,759 DadaGP tokens to the tokenizer, resizes embeddings,
                 and trains embed_tokens + lm_head (via modules_to_save)
                 alongside the LoRA adapter. Reports the achieved
                 DadaGP-token -> BPE compression ratio and peak VRAM —
                 the two numbers that decide option A vs B.

Usage:
  python src/train_qlora.py --smoke-test
  python src/train_qlora.py --smoke-test --extend-vocab
"""
import argparse, functools, glob, json, os, random

import numpy as np
import torch
from tqdm import tqdm
from datasets import Dataset
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForLanguageModeling,
                          EarlyStoppingCallback, Trainer, TrainerCallback,
                          TrainingArguments)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_token_texts(paths: list[str]) -> list[str]:
    """Accepts raw *.tokens.txt files or prepared *.jsonl (one song per record)."""
    texts = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            if p.endswith(".jsonl"):
                texts.extend(json.loads(line)["text"] for line in f if line.strip())
            else:
                texts.append(f.read())
    return texts


def sample_jsonl_texts(path: str, n: int, seed: int) -> list[str]:
    """Sample n songs from a prepared JSONL without holding the file in RAM.
    Two passes: count lines, then parse only the sampled ones. n=0 -> all."""
    with open(path, encoding="utf-8") as f:
        total = sum(1 for line in f if line.strip())
    take = set(range(total)) if not n or n >= total else \
        set(random.Random(seed).sample(range(total), n))
    texts, i = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if i in take:
                texts.append(json.loads(line)["text"])
            i += 1
    print(f"[data] {len(texts)}/{total} songs from {os.path.basename(path)}")
    return texts


def chunk_examples(texts: list[str], tokenizer, seq_len: int,
                   max_windows: int = 0, cache: str = None) -> Dataset:
    """Tokenize whole songs, slice into fixed-length training windows.
    cache: optional .npy path — full-corpus tokenization is ~an hour, and a
    pod restart shouldn't have to repeat it."""
    if cache and os.path.exists(cache):
        windows = list(np.load(cache, allow_pickle=True))
        print(f"[data] {len(windows)} windows loaded from cache {cache}")
        return Dataset.from_dict({"input_ids": windows})
    windows = []
    for t in tqdm(texts, desc="tokenize", unit="song"):
        ids = tokenizer(t, add_special_tokens=False).input_ids
        for i in range(0, max(len(ids) - seq_len, 1), seq_len):
            windows.append(np.asarray(ids[i:i + seq_len], dtype=np.int32))
            if max_windows and len(windows) >= max_windows:
                return Dataset.from_dict({"input_ids": windows})
    if cache:
        np.save(cache, np.asarray(windows, dtype=object), allow_pickle=True)
        print(f"[data] window cache saved -> {cache}")
    return Dataset.from_dict({"input_ids": windows})


class HubPushCallback(TrainerCallback):
    """Push each finished checkpoint to a private HF repo. On a cloud pod the
    disk dies with the pod — the Hub copy is what survives a preemption."""

    def __init__(self, repo_id: str):
        self.api = HfApi()
        self.repo = repo_id
        self.api.create_repo(repo_id, private=True, exist_ok=True)

    def on_save(self, args, state, control, **kwargs):
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt):
            self.api.upload_folder(
                repo_id=self.repo, folder_path=ckpt,
                path_in_repo=f"checkpoint-{state.global_step}",
                run_as_future=True)  # non-blocking; training continues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--data-glob",
                    default=os.path.join(REPO_ROOT, "data", "prepared", "train.jsonl"),
                    help="prepared train.jsonl (from prepare_data.py) or a glob of *.tokens.txt")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--pilot-songs", type=int, default=0,
                    help="sample this many training songs (0 = all)")
    ap.add_argument("--val-jsonl",
                    default=os.path.join(REPO_ROOT, "data", "prepared", "val.jsonl"))
    ap.add_argument("--val-songs", type=int, default=100)
    ap.add_argument("--eval-windows", type=int, default=64)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extend-vocab", action="store_true",
                    help="add DadaGP tokens to the tokenizer and train embeddings")
    ap.add_argument("--new-tokens-only", action="store_true",
                    help="with --extend-vocab: train only the added embedding rows "
                         "(~8M params) instead of full embed_tokens+lm_head copies")
    ap.add_argument("--vocab-json",
                    default=os.path.join(REPO_ROOT, "data", "corpus", "DadaGP-v1.1",
                                         "_DadaGP_all_tokens.json"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "outputs", "qlora"))
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint in --out")
    ap.add_argument("--hub-repo", default=None,
                    help="private HF model repo id (user/name); every checkpoint "
                         "and the final adapter are pushed there")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-scheduler", default="constant",
                    help="constant for short pilots; cosine for full runs")
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument("--window-cache", default=None,
                    help=".npy path to cache tokenized training windows")
    args = ap.parse_args()

    if args.smoke_test:
        paths = glob.glob(os.path.join(REPO_ROOT, "data", "samples", "*.tokens.txt"))
        suffix = "-extvocab" if args.extend_vocab else ""
        args.out = os.path.join(REPO_ROOT, "outputs", f"smoke{suffix}")
    else:
        paths = glob.glob(args.data_glob)
    if not paths:
        raise SystemExit("no training files found")
    print(f"[data] {len(paths)} file(s): {[os.path.basename(p) for p in paths]}")

    assert torch.cuda.is_available(), "CUDA required"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map={"": 0})

    modules_to_save = None
    trainable_token_indices = None
    if args.extend_vocab:
        with open(args.vocab_json, encoding="utf-8") as f:
            dadagp_vocab = json.load(f)
        n_added = tok.add_tokens(dadagp_vocab)
        model.resize_token_embeddings(len(tok))
        if args.new_tokens_only:
            trainable_token_indices = {
                "embed_tokens": list(range(len(tok) - n_added, len(tok)))}
        else:
            modules_to_save = ["embed_tokens", "lm_head"]
        print(f"[vocab] added {n_added}/{len(dadagp_vocab)} DadaGP tokens "
              f"-> tokenizer size {len(tok)}")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=modules_to_save,
        trainable_token_indices=trainable_token_indices,
        # Qwen ties embed_tokens/lm_head; without this PEFT trains two
        # separate 311M copies — the difference between fitting 8GB or not
        ensure_weight_tying=modules_to_save is not None,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    if args.extend_vocab:
        # PEFT's save_embedding_layers="auto" sees the resized vocab and adds
        # full embed_tokens+lm_head copies to every save (~2.6 GB per
        # checkpoint; the disk spike that coincided with the 2026-07-20 BSOD).
        # Redundant: trainable-token rows are stored as full replacement
        # values inside the adapter, so reload only needs base model + resize.
        model.save_pretrained = functools.partial(
            model.save_pretrained, save_embedding_layers=False)

    if args.smoke_test:
        texts = load_token_texts(paths)
    else:
        texts = sample_jsonl_texts(paths[0], args.pilot_songs, args.seed)
    if args.extend_vocab and args.smoke_test:
        n_words = sum(len(t.split()) for t in texts)
        n_bpe = sum(len(tok(t, add_special_tokens=False).input_ids) for t in texts)
        print(f"[vocab] compression on training text: {n_words} DadaGP tokens -> "
              f"{n_bpe} BPE ids ({n_bpe / n_words:.2f} ids per DadaGP token; "
              f"stock tokenizer was ~8.0)")
    ds = chunk_examples(texts, tok, args.seq_len, cache=args.window_cache)
    print(f"[data] {len(ds)} training windows of {args.seq_len} tokens")

    eval_ds = None
    if not args.smoke_test and os.path.exists(args.val_jsonl):
        vtexts = sample_jsonl_texts(args.val_jsonl, args.val_songs, args.seed)
        eval_ds = chunk_examples(vtexts, tok, args.seq_len,
                                 max_windows=args.eval_windows)
        print(f"[data] {len(eval_ds)} eval windows")
    do_eval = eval_ds is not None

    targs = TrainingArguments(
        output_dir=args.out,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        logging_steps=5,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to=[],
        eval_strategy="steps" if do_eval else "no",
        eval_steps=args.eval_steps,
        prediction_loss_only=True,   # never gather 155k-wide logits on eval
        save_strategy="steps" if do_eval else "no",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=do_eval,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] if do_eval else []
    hub = HubPushCallback(args.hub_repo) if args.hub_repo else None
    if hub:
        callbacks.append(hub)
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds, eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
        callbacks=callbacks,
    )
    result = trainer.train(resume_from_checkpoint=args.resume or None)

    losses = [l["loss"] for l in trainer.state.log_history if "loss" in l]
    evals = [(l["step"], l["eval_loss"]) for l in trainer.state.log_history
             if "eval_loss" in l]
    print(f"\n[smoke] loss first->last: {losses[0]:.3f} -> {losses[-1]:.3f}"
          if args.smoke_test and losses else f"\n[train] final loss: {losses[-1]:.3f}")
    for step, el in evals:
        print(f"[eval] step {step}: val_loss {el:.4f}")
    print(f"[vram] peak allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    model.save_pretrained(args.out)
    if args.extend_vocab:
        tok.save_pretrained(args.out)  # extended tokenizer is part of the artifact
    print(f"adapter saved -> {args.out}")
    if hub:
        hub.api.upload_folder(repo_id=hub.repo, folder_path=args.out,
                              path_in_repo="final",
                              ignore_patterns=["checkpoint-*"])
        print(f"adapter pushed -> hf.co/{hub.repo} (final/)")


if __name__ == "__main__":
    main()
