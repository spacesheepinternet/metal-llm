"""
QLoRA fine-tune of a small instruct model on DadaGP token text.

Two modes:
  --smoke-test : tiny overfit run on the two sample songs (~40 steps).
                 Proves the full 4-bit + LoRA training loop works locally
                 (Windows + bitsandbytes + 8GB VRAM) before the corpus lands.
                 Pass criterion: training loss drops hard (memorization).
  (default)    : real training over data/corpus/ — wired up once the
                 DadaGP corpus arrives.

Usage:
  python src/train_qlora.py --smoke-test
"""
import argparse, glob, os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_token_texts(paths: list[str]) -> list[str]:
    texts = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            texts.append(f.read())
    return texts


def chunk_examples(texts: list[str], tokenizer, seq_len: int) -> Dataset:
    """Tokenize whole songs, slice into fixed-length training windows."""
    windows = []
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False).input_ids
        for i in range(0, max(len(ids) - seq_len, 1), seq_len):
            windows.append(ids[i:i + seq_len])
    return Dataset.from_dict({"input_ids": windows})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--data-glob", default=os.path.join(REPO_ROOT, "data", "corpus", "*.txt"))
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "outputs", "qlora"))
    args = ap.parse_args()

    if args.smoke_test:
        paths = glob.glob(os.path.join(REPO_ROOT, "data", "samples", "*.tokens.txt"))
        args.out = os.path.join(REPO_ROOT, "outputs", "smoke")
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
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = chunk_examples(load_token_texts(paths), tok, args.seq_len)
    print(f"[data] {len(ds)} windows of {args.seq_len} tokens")

    targs = TrainingArguments(
        output_dir=args.out,
        max_steps=args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        warmup_steps=2,
        logging_steps=5,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to=[],
        save_strategy="no",
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    result = trainer.train()

    losses = [l["loss"] for l in trainer.state.log_history if "loss" in l]
    print(f"\n[smoke] loss first->last: {losses[0]:.3f} -> {losses[-1]:.3f}"
          if args.smoke_test and losses else f"\n[train] final loss: {losses[-1]:.3f}")
    print(f"[vram] peak allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    model.save_pretrained(args.out)
    print(f"adapter saved -> {args.out}")


if __name__ == "__main__":
    main()
