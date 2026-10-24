"""
MP-G01: Poetry Line Completion Assistant
Continued pretraining of GPT-2 (124M) on a public-domain poetry corpus.

Expected input:
  data/poetry.txt  - plain-text corpus containing public-domain poems.
The corpus should be around 5–10 MB as specified in the project allocation.

Run:
  python poetry_assistant.py train
  python poetry_assistant.py evaluate
  python poetry_assistant.py app

Important:
- The program records real measurements; do not write placeholder results in the report.
- GPT-2's token IDs are compatible with the GPT-2 tiktoken encoding.
- This implementation loads pretrained GPT-2 weights through Hugging Face Transformers.
"""

import argparse
import csv
import json
import math
import os
import random
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Config

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoint"
CORPUS_PATH = DATA_DIR / "poetry.txt"
BASE_MODEL_NAME = "gpt2"  # GPT-2 small, approximately 124M parameters
ENCODING_NAME = "gpt2"

# Adjust these settings to match the available hardware.
CONTEXT_LENGTH = 256
STRIDE = 128
BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
VALIDATION_FRACTION = 0.10
RANDOM_SEED = 123
MAX_NEW_TOKENS = 80
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_and_clean_corpus(path=CORPUS_PATH):
    """Read plain text, remove common Gutenberg boilerplate, and preserve poem line breaks."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Corpus not found at {path}. Put your public-domain poetry text in data/poetry.txt."
        )
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    original_chars = len(text)

    # Remove common Project Gutenberg start/end markers if present.
    start_patterns = [
        r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
    ]
    end_patterns = [
        r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*",
        r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*",
    ]
    for pattern in start_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            text = text[match.end():]
            break
    for pattern in end_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            text = text[:match.start()]
            break

    # Normalize newlines and whitespace without flattening stanza structure.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise ValueError("The corpus is empty after cleaning. Check data/poetry.txt.")

    print(f"Characters before cleaning: {original_chars:,}")
    print(f"Characters after cleaning:  {len(text):,}")
    print(f"Approximate text size:      {len(text.encode('utf-8')) / (1024 * 1024):.2f} MB")
    return text


class SlidingWindowDataset(Dataset):
    """Overlapping next-token training windows, created separately per data split."""
    def __init__(self, token_ids, context_length=CONTEXT_LENGTH, stride=STRIDE):
        self.token_ids = token_ids
        self.context_length = context_length
        self.stride = stride
        self.starts = list(range(0, max(0, len(token_ids) - context_length), stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        chunk = self.token_ids[start:start + self.context_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def make_loaders(text, tokenizer, context_length=CONTEXT_LENGTH, stride=STRIDE,
                 batch_size=BATCH_SIZE):
    # Split by text before windowing to reduce train/validation leakage.
    split_at = int(len(text) * (1 - VALIDATION_FRACTION))
    train_text = text[:split_at]
    val_text = text[split_at:]
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    train_ds = SlidingWindowDataset(train_ids, context_length, stride)
    val_ds = SlidingWindowDataset(val_ids, context_length, stride)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            "Not enough tokens to make sliding windows. Use a larger corpus or reduce CONTEXT_LENGTH."
        )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    print(f"Training tokens: {len(train_ids):,}; validation tokens: {len(val_ids):,}")
    print(f"Training windows: {len(train_ds):,}; validation windows: {len(val_ds):,}")
    return train_loader, val_loader, train_ids, val_ids, train_text, val_text


def load_model():
    """Load pretrained GPT-2 small weights (124M) and move the model to the selected device."""
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_NAME)
    model.config.use_cache = False
    model.to(DEVICE)
    return model


@torch.no_grad()
def evaluate_loss(model, loader):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(input_ids=x, labels=y)
        total_loss += float(output.loss.item())
        total_batches += 1
    model.train()
    return total_loss / max(total_batches, 1)


def perplexity(loss):
    # Clamp only to avoid numeric overflow in display.
    return math.exp(min(float(loss), 20.0))


def save_training_plot(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o", label="Training loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GPT-2 Continued Pretraining: Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = OUTPUT_DIR / "loss_curve.png"
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"Saved loss plot: {path}")


def train():
    set_seed()
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Please place the corpus at {CORPUS_PATH}. See README.md for setup."
        )
    raw_text = read_and_clean_corpus()
    tokenizer = tiktoken.get_encoding(ENCODING_NAME)
    train_loader, val_loader, train_ids, val_ids, train_text, val_text = make_loaders(raw_text, tokenizer)

    model = load_model()
    baseline_val_loss = evaluate_loss(model, val_loader)
    baseline_val_ppl = perplexity(baseline_val_loss)
    print(f"Before training - validation loss: {baseline_val_loss:.4f}, perplexity: {baseline_val_ppl:.2f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_losses, val_losses = [], []
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []
        for step, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            output = model(input_ids=x, labels=y)
            loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))
            if step % 50 == 0:
                print(f"Epoch {epoch + 1}/{EPOCHS}, batch {step}/{len(train_loader)}, loss={loss.item():.4f}")

        train_loss = sum(batch_losses) / max(len(batch_losses), 1)
        val_loss = evaluate_loss(model, val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch {epoch + 1}: train loss={train_loss:.4f}, "
            f"validation loss={val_loss:.4f}, validation perplexity={perplexity(val_loss):.2f}"
        )

    elapsed = time.time() - start_time
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer_name = "tiktoken gpt2 encoding"
    save_training_plot(train_losses, val_losses)

    results = {
        "base_model": BASE_MODEL_NAME,
        "device": str(DEVICE),
        "corpus_path": str(CORPUS_PATH),
        "corpus_bytes": len(raw_text.encode("utf-8")),
        "context_length": CONTEXT_LENGTH,
        "stride": STRIDE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "train_windows": len(train_loader.dataset),
        "validation_windows": len(val_loader.dataset),
        "baseline_validation_loss": baseline_val_loss,
        "baseline_validation_perplexity": baseline_val_ppl,
        "final_validation_loss": val_losses[-1],
        "final_validation_perplexity": perplexity(val_losses[-1]),
        "train_losses": train_losses,
        "validation_losses": val_losses,
        "training_time_seconds": elapsed,
        "tokenizer": tokenizer_name,
    }
    (OUTPUT_DIR / "training_results.json").write_text(json.dumps(results, indent=2))
    print(f"Training time: {elapsed / 60:.2f} minutes")
    print(f"Saved checkpoint to: {CHECKPOINT_DIR}")
    print(f"Saved metrics to: {OUTPUT_DIR / 'training_results.json'}")
    print("Training complete. Run: python poetry_assistant.py evaluate")


def get_model_and_tokenizer(require_checkpoint=True):
    tokenizer = tiktoken.get_encoding(ENCODING_NAME)
    if require_checkpoint and (not CHECKPOINT_DIR.exists() or not (CHECKPOINT_DIR / "config.json").exists()):
        raise FileNotFoundError("Checkpoint not found. Run `python poetry_assistant.py train` first.")
    if (CHECKPOINT_DIR / "config.json").exists():
        model = GPT2LMHeadModel.from_pretrained(CHECKPOINT_DIR)
    else:
        model = load_model()
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS,
                  strategy="top_k", temperature=0.8, top_k=40):
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Please enter a poem beginning.")
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)
    if input_ids.shape[1] == 0:
        raise ValueError("The prompt did not produce any tokens.")

    generated = input_ids
    for _ in range(max_new_tokens):
        # GPT-2 has a maximum context length of 1024 tokens.
        context = generated[:, -1024:]
        logits = model(input_ids=context).logits[:, -1, :]

        if strategy == "greedy":
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            temp = max(float(temperature), 1e-5)
            logits = logits / temp
            if strategy == "top_k":
                k = min(int(top_k), logits.shape[-1])
                top_values, _ = torch.topk(logits, k)
                cutoff = top_values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_id], dim=1)
        # Stop on end-of-text if the model emits it.
        if int(next_id.item()) == 50256:
            break

    new_tokens = generated[0, input_ids.shape[1]:].tolist()
    return tokenizer.decode(new_tokens).strip()


def split_generated_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def normalize_line(line):
    line = line.lower().strip()
    line = re.sub(r"\s+", " ", line)
    line = re.sub(r"^[\W_]+|[\W_]+$", "", line)
    return line


def evaluate_generation_and_copying():
    model, tokenizer = get_model_and_tokenizer()
    raw_text = read_and_clean_corpus()
    training_lines = {
        normalize_line(line) for line in raw_text.splitlines()
        if len(normalize_line(line)) >= 8
    }
    prompts = [
        "The moonlight rests upon the lake",
        "A lonely bird begins to sing",
        "The autumn leaves are drifting down",
        "Beyond the hills, the morning glows",
        "I kept a dream inside my heart",
    ]
    strategies = [
        ("greedy", 1.0, 0),
        ("temperature_0.8", 0.8, 0),
        ("temperature_1.2", 1.2, 0),
        ("top_k_40", 0.8, 40),
    ]
    rows = []
    generated_lines = []
    for prompt in prompts:
        for label, temp, k in strategies:
            strategy = "greedy" if label == "greedy" else ("top_k" if k else "temperature")
            continuation = generate_text(
                model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS,
                strategy=strategy, temperature=temp, top_k=(k or 40)
            )
            # Store only the generated continuation, not the supplied prompt.
            lines = split_generated_lines(continuation)
            for line in lines:
                generated_lines.append(line)
            rows.append({
                "prompt": prompt,
                "strategy": label,
                "temperature": temp,
                "top_k": k if k else "",
                "continuation": continuation.replace("\n", " | "),
                "exact_line_match": any(normalize_line(line) in training_lines for line in lines),
            })
            print(f"\n[{label}] Prompt: {prompt}\n{continuation}\n")

    total_lines = len(generated_lines)
    copied_lines = sum(1 for line in generated_lines if normalize_line(line) in training_lines)
    copied_pct = (100.0 * copied_lines / total_lines) if total_lines else 0.0
    csv_path = OUTPUT_DIR / "generation_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "generated_nonempty_lines": total_lines,
        "exact_matching_lines": copied_lines,
        "exact_line_copy_percentage": copied_pct,
        "note": "Exact normalized line matches only; does not detect paraphrases or near-duplicates.",
    }
    (OUTPUT_DIR / "copying_check.json").write_text(json.dumps(summary, indent=2))
    print(f"\nExact matching lines: {copied_lines}/{total_lines} ({copied_pct:.2f}%)")
    print(f"Saved generations to {csv_path}")
    print(f"Saved copying summary to {OUTPUT_DIR / 'copying_check.json'}")


def launch_app():
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError("Install Gradio with `pip install gradio`.") from exc

    model, tokenizer = get_model_and_tokenizer()
    def next_lines(poem_so_far, n=2, decoding="Top-k (k=40)", temperature=0.8):
        strategy_map = {
            "Greedy": ("greedy", 1.0, 40),
            "Temperature": ("temperature", float(temperature), 40),
            "Top-k (k=40)": ("top_k", float(temperature), 40),
        }
        strategy, temp, k = strategy_map[decoding]
        # Generate a modest continuation, then show the first n non-empty lines.
        continuation = generate_text(
            model, tokenizer, poem_so_far, max_new_tokens=MAX_NEW_TOKENS,
            strategy=strategy, temperature=temp, top_k=k
        )
        lines = split_generated_lines(continuation)
        if len(lines) < int(n):
            # Keep output usable even when the model emits a single long line.
            lines = lines + ["" for _ in range(int(n) - len(lines))]
        return "\n".join(lines[:int(n)])

    demo = gr.Interface(
        fn=next_lines,
        inputs=[
            gr.Textbox(lines=5, label="Poem so far", placeholder="Enter the beginning of your poem..."),
            gr.Slider(minimum=1, maximum=4, value=2, step=1, label="Number of suggested lines"),
            gr.Dropdown(["Greedy", "Temperature", "Top-k (k=40)"], value="Top-k (k=40)", label="Decoding method"),
            gr.Slider(minimum=0.5, maximum=1.5, value=0.8, step=0.1, label="Temperature (for sampling methods)"),
        ],
        outputs=gr.Textbox(lines=6, label="Suggested continuation"),
        title="Poetry Line Completion Assistant",
        description="A GPT-2-based creative-writing assistant. Suggestions may be repetitive or resemble training text.",
    )
    demo.launch()


def main():
    parser = argparse.ArgumentParser(description="MP-G01 Poetry Line Completion Assistant")
    parser.add_argument("command", choices=["train", "evaluate", "app"], help="Action to run")
    args = parser.parse_args()
    if args.command == "train":
        train()
    elif args.command == "evaluate":
        evaluate_generation_and_copying()
    elif args.command == "app":
        launch_app()


if __name__ == "__main__":
    main()
