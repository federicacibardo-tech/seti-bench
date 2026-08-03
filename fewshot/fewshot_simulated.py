"""Few-shot classification of simulated cadences (candidate vs RFI).

Uses an explicit counting procedure: the model counts the distinct tracks, checks
each one against the OFF panels, and decides. This is more robust than a single
holistic judgement on multi-track cadences. Runs on Ollama with three in-context
examples (one pure candidate, one mixed candidate, one RFI).
"""

import json
import random

import ollama

LABELS_FILE = "labels.jsonl"
RESULTS_FILE = "results_fewshot_simulated.jsonl"
MODEL = "qwen3.5:9b"
LIMIT = 250

CLASSES = ["candidate", "rfi"]

EXAMPLE_PURE = "pure_candidate_000.png"
EXAMPLE_MIXED = "mixed_candidate_000.png"
EXAMPLE_RFI = "rfi_only_000.png"
EXAMPLE_FILES = [EXAMPLE_PURE, EXAMPLE_MIXED, EXAMPLE_RFI]

PROMPT = (
    "You are a brightness detector for a 6-panel spectrogram (1 to 6 from top to bottom).\n"
    "- Panels 1, 3, 5 are ON-target.\n"
    "- Panels 2, 4, 6 are OFF-target.\n\n"
    "A signal is a localized, bright track (straight, tilted, or wavy) that stands out "
    "against the background. An image can contain one or several tracks at different "
    "horizontal positions (frequencies).\n\n"
    "FOLLOW THIS PROCEDURE:\n"
    "1. Count how many distinct bright tracks you see (look across the whole width).\n"
    "2. For EACH track, check its horizontal position: does a track exist at that SAME "
    "position in the OFF panels (2, 4, 6)?\n"
    "   - track present in ON and also in OFF at that position = RFI track\n"
    "   - track present in ON but NOT in OFF at that position = ON-only track\n"
    "3. Decide:\n"
    "   - CLASS 'candidate': at least ONE track is ON-only. (Even if OTHER tracks are "
    "RFI present everywhere - one ON-only track is enough to make it a candidate.)\n"
    "   - CLASS 'rfi': there is only interference - every track appears in all 6 panels, "
    "and NO track is ON-only.\n\n"
    "IMPORTANT: if there is only ONE track and it appears in all panels, it is rfi. "
    "Do not call it candidate just because it looks a bit brighter in some ON panels.\n\n"
    "Answer immediately.\n"
    "Format your output exactly as follows (no markdown, no code blocks):\n"
    "REASON: [how many tracks, and is any of them ON-only?]\n"
    "CLASS: [Write exactly one of: candidate or rfi]"
)

ANSWER_PURE = (
    "REASON: One track, visible only in the ON panels (1, 3, 5) and absent from the OFF "
    "panels, so it is ON-only -> candidate.\nCLASS: candidate"
)
ANSWER_MIXED = (
    "REASON: Two tracks: one appears in all panels (RFI), the other appears only in the "
    "ON panels and not in the OFF panels (ON-only), so it is a candidate.\nCLASS: candidate"
)
ANSWER_RFI = (
    "REASON: One track, present in all 6 panels including the OFF ones; no ON-only "
    "track, so it is interference.\nCLASS: rfi"
)


def classify(path):
    messages = [
        {"role": "user", "content": PROMPT, "images": [EXAMPLE_PURE]},
        {"role": "assistant", "content": ANSWER_PURE},
        {"role": "user", "content": PROMPT, "images": [EXAMPLE_MIXED]},
        {"role": "assistant", "content": ANSWER_MIXED},
        {"role": "user", "content": PROMPT, "images": [EXAMPLE_RFI]},
        {"role": "assistant", "content": ANSWER_RFI},
        {"role": "user", "content": PROMPT, "images": [path]},
    ]
    response = ollama.chat(model=MODEL, messages=messages, think=False,
                           options={"num_ctx": 16384, "num_predict": 150, "temperature": 0})
    return response["message"]["content"]


def parse(text):
    if not text:
        return "UNCLEAR"
    t = text.lower()
    if "</think>" in t:
        t = t.split("</think>")[-1]
    for line in t.splitlines():
        if "class:" in line:
            for c in CLASSES:
                if c in line:
                    return c
    found = [c for c in CLASSES if c in t]
    return found[0] if len(found) == 1 else "UNCLEAR"


def label_of(record):
    return "candidate" if record["has_candidate"] else "rfi"


def main():
    records = []
    with open(LABELS_FILE) as f:
        for line in f:
            records.append(json.loads(line))
    records = [r for r in records if r["filename"] not in EXAMPLE_FILES]

    random.seed(0)
    random.shuffle(records)
    records = records[:LIMIT]

    correct = total = 0
    matrix = {v: {p: 0 for p in CLASSES + ["UNCLEAR"]} for v in CLASSES}

    print(f"Classifying {len(records)} images with {MODEL}\n", flush=True)

    with open(RESULTS_FILE, "w") as out:
        for i, record in enumerate(records, 1):
            filename = record["filename"]
            true_class = label_of(record)
            response = classify(filename)
            pred = parse(response)

            ok = pred == true_class
            if ok:
                correct += 1
            total += 1
            if pred in matrix[true_class]:
                matrix[true_class][pred] += 1

            print(f"[{i}/{len(records)}] {filename} true={true_class} -> {pred} "
                  f"({'OK' if ok else 'X'})  {correct}/{total}", flush=True)

            out.write(json.dumps({
                "filename": filename, "true_class": true_class,
                "pred_class": pred, "response": response, "correct": ok,
            }) + "\n")

    print("\n" + "=" * 50)
    print(f"Accuracy: {correct}/{total} = {correct / total * 100:.1f}%")
    print("=" * 50)
    for c in CLASSES:
        n = sum(matrix[c].values())
        if n:
            print(f"  {c:12s}: {matrix[c][c]}/{n} ({matrix[c][c] / n * 100:.0f}%)")


if __name__ == "__main__":
    main()
