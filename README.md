# MP-G01 — Poetry Line Completion Assistant

This project continues pretraining GPT-2 small (124M) on public-domain poetry text and provides a small Gradio application that suggests continuation lines.

## 1. Project files

- `poetry_assistant.py` — preprocessing, sliding-window dataset/DataLoader, training, evaluation, decoding comparison, exact-line copying check, and Gradio application.
- `data/poetry.txt` — **you must supply** the poetry corpus assigned by your course (target size 5–10 MB).
- `checkpoint/` — saved model checkpoint created after training.

## 2. Environment setup

Use Python 3.10 or 3.11 if possible.

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first training run downloads the pretrained GPT-2 small weights from Hugging Face. A network connection is required for that download.

## 3. Add the dataset

Create the `data` directory if it does not exist and save the course-approved public-domain poetry corpus as:

`data/poetry.txt`

Use the exact dataset/source specified by your instructor. Record its URL, license/terms, source citation, file size, and how you selected the 5–10 MB subset. Do not claim the included program downloaded or verified a dataset automatically.

## 4. Train

```bash
python poetry_assistant.py train
```

The code uses GPT-2's pretrained weights, the GPT-2 `tiktoken` encoding, a sliding-window next-token dataset, and AdamW. It records baseline validation loss/perplexity, trains for the configured number of epochs, evaluates again, saves a checkpoint, and creates `outputs/loss_curve.png`.

For a small assignment run, the defaults are `CONTEXT_LENGTH=256`, `STRIDE=128`, `BATCH_SIZE=2`, `EPOCHS=1`. Change these only if you have enough memory/time, and report the actual values used. If GPU memory is limited, reduce batch size or context length.

## 5. Evaluate decoding and copying

```bash
python poetry_assistant.py evaluate
```

This compares greedy decoding, temperature 0.8, temperature 1.2, and top-k (k=40) on the same prompts. It saves real generations to a CSV and computes the percentage of generated non-empty lines that exactly match a normalized line in the corpus. This is only an exact-line check; it does not detect paraphrases or near-duplicates.

## 6. Launch the application

```bash
python poetry_assistant.py app
```

Open the local Gradio URL printed in the terminal. Enter a poem beginning, choose the number of lines and decoding method, and click Submit/Generate.

## 7. Report checklist

Use real output values from `outputs/training_results.json` and `outputs/copying_check.json`. Do not make up metrics. Add the generated comparison examples from `outputs/generation_comparison.csv` to the report and capture screenshots of the actual running application.
