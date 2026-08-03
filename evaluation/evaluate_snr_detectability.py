"""Measure detection rate versus SNR, with a false-positive control.

For each SNR level the recovery rate is the fraction of candidates the model
flags as present. The pure-noise images give the false-positive rate: a model
that answers "candidate" indiscriminately reaches 100% recovery but also flags
noise, so recovery must be read together with the false-positive rate.

Set MODEL to evaluate each Ollama model in turn.
"""

import json
from collections import defaultdict

import ollama

LABELS_FILE = "labels_snrtest.jsonl"
RESULTS_FILE = "results_snr_detection.jsonl"
MODEL = "qwen3.5:9b"

CLASSES = ["candidate", "noise_only"]

records = []
with open(LABELS_FILE) as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

# few-shot example: a clearly visible high-SNR candidate
high_snr = [r for r in records if r["class"] == "candidate" and r["params"].get("snr") == 25]
EXAMPLE = high_snr[-1]["filename"] if high_snr else records[0]["filename"]
records = [r for r in records if r["filename"] != EXAMPLE]

PROMPT = (
    "You are a brightness detector for a 6-panel spectrogram (1 to 6 from top to bottom).\n"
    "- Panels 1, 3, 5 are ON-target.\n"
    "- Panels 2, 4, 6 are OFF-target.\n\n"
    "A signal is a thin track against a noisy background; it can be extremely faint and "
    "very thin. Be cautious: a faint track and the noisy background look similar, so do "
    "not confuse them.\n"
    "   - CLASS 'candidate': a thin track is present in the ON panels (1, 3, 5).\n"
    "   - CLASS 'noise_only': all panels are clean noise, no track.\n\n"
    "Inspect the ON panels (1, 3, 5): a thin track -> candidate; only speckled noise -> "
    "noise_only. Do NOT jump to conclusions: only call it candidate if you actually see "
    "a thin line standing against the background, not just bright noise.\n\n"
    "Format:\n"
    "REASON: [one short sentence about the ON panels]\n"
    "CLASS: [candidate or noise_only]"
)

EXAMPLE_ANSWER = ("REASON: A thin track is visible in the ON panels (1, 3, 5).\n"
                  "CLASS: candidate")


def classify(path):
    messages = [
        {"role": "user", "content": PROMPT, "images": [EXAMPLE]},
        {"role": "assistant", "content": EXAMPLE_ANSWER},
        {"role": "user", "content": PROMPT, "images": [path]},
    ]
    response = ollama.chat(model=MODEL, messages=messages, think=False,
                           options={"num_ctx": 16384, "num_predict": 100, "temperature": 0})
    return response["message"]["content"]


def parse(text):
    if not text:
        return "UNCLEAR"
    t = text.lower().replace("noise only", "noise_only")
    if "</think>" in t:
        t = t.split("</think>")[-1]
    for line in t.splitlines():
        if "class:" in line:
            for c in CLASSES:
                if c in line:
                    return c
    found = [c for c in CLASSES if c in t]
    return found[0] if len(found) == 1 else "UNCLEAR"


per_snr = defaultdict(lambda: [0, 0])     # snr -> [detected, total]
noise_fp = [0, 0]                          # [false positives, total]

print(f"Model: {MODEL}  |  evaluating {len(records)} images\n", flush=True)

with open(RESULTS_FILE, "w") as out:
    for i, record in enumerate(records, 1):
        filename = record["filename"]
        true_class = record["class"]
        response = classify(filename)
        pred = parse(response)

        if true_class == "candidate":
            snr = record["params"]["snr"]
            detected = pred == "candidate"
            per_snr[snr][1] += 1
            if detected:
                per_snr[snr][0] += 1
            print(f"[{i}] {filename} SNR={snr:.0f} -> {pred} "
                  f"[{'detected' if detected else 'missed'}]", flush=True)
        else:
            false_positive = pred == "candidate"
            noise_fp[1] += 1
            if false_positive:
                noise_fp[0] += 1
            print(f"[{i}] {filename} NOISE -> {pred} "
                  f"[{'FALSE POSITIVE' if false_positive else 'ok'}]", flush=True)

        out.write(json.dumps({
            "filename": filename, "true_class": true_class,
            "snr": record["params"].get("snr"), "prediction": pred,
            "response": response,
        }) + "\n")

print("\n" + "=" * 52)
print(f" DETECTABILITY vs SNR  -  {MODEL}")
print("=" * 52)
print(f"{'SNR':>6} | {'Detected/Total':>14} | {'Recovery':>10}")
print("-" * 52)
for snr in sorted(per_snr):
    detected, total = per_snr[snr]
    print(f"{snr:6.0f} | {detected:6d}/{total:<6d} | {detected / total * 100:9.0f}%")
print("-" * 52)
fp, total = noise_fp
print(f"False positives on noise_only: {fp}/{total} = {fp / total * 100:.0f}%")
print("=" * 52)
