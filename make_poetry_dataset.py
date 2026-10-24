import gzip
import json
from pathlib import Path

source = Path("data/poetry_corpus.ndjson.gz")
output = Path("data/poetry.txt")
target_size = 8 * 1024 * 1024  # approximately 8 MB

total_bytes = 0
line_count = 0

with gzip.open(source, "rt", encoding="utf-8") as infile:
    with output.open("w", encoding="utf-8") as outfile:
        for record in infile:
            try:
                poem_line = json.loads(record).get("s", "").strip()
            except (json.JSONDecodeError, AttributeError):
                continue

            if not poem_line:
                continue

            text = poem_line + "\n"
            size = len(text.encode("utf-8"))

            if total_bytes + size > target_size:
                break

            outfile.write(text)
            total_bytes += size
            line_count += 1

print(f"Created: {output}")
print(f"Poetry lines: {line_count}")
print(f"File size: {total_bytes / (1024 * 1024):.2f} MB")
