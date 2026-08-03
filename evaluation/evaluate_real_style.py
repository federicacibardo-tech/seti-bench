"""Evaluate the fine-tuned model on real-style images with ground truth.

The real-style set is generated with known labels, so accuracy can be measured
directly. A high score here with a low score on genuine real data isolates the
failure to plot rendering (domain shift) rather than signal content.
Run inside the NGC container.
"""

import os
import json
import re

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from PIL import Image
from unsloth import FastVisionModel

torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True

IMAGE_DIR = "/workspace"
LABELS_FILE = "labels_realstyle.jsonl"
MODEL_DIR = "qwen35_finetuned"
RESULTS_FILE = "results_realstyle.jsonl"

PROMPT = (
    "You are analyzing a 6-panel cadence spectrogram (panels 1 to 6, top to bottom).\n"
    "- ON-target panels: 1, 3, 5.\n"
    "- OFF-target panels: 2, 4, 6.\n\n"
    "A signal is a thin bright track (straight, tilted, or wavy). An image may contain "
    "one or more tracks at different frequency positions.\n"
    "- A CANDIDATE is a thin track present ONLY in the ON panels (1, 3, 5), absent from "
    "the OFF panels. It can be faint. A cadence has a candidate if at least one such "
    "ON-only track exists, EVEN IF an RFI track is also present.\n"
    "- RFI is any track present in ALL six panels (ON and OFF).\n\n"
    "Report whether a candidate is present, and describe any RFI track you see.\n"
    "Format your answer exactly as:\n"
    "CANDIDATE: [yes or no]\n"
    "RFI: [short description of the RFI track, or 'none']"
)

print("Loading fine-tuned model...")
model, processor = FastVisionModel.from_pretrained(MODEL_DIR, load_in_4bit=True)
FastVisionModel.for_inference(model)


def classify(path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((1400, 1400))
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": PROMPT},
    ]}]
    try:
        text = processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=img, text=text, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=128, do_sample=False, use_cache=True)
    return processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def parse(text):
    candidate, rfi = None, ""
    if not text:
        return candidate, rfi
    t = text
    if "</think>" in t.lower():
        t = t[t.lower().rindex("</think>") + len("</think>"):]
    m = re.search(r"CANDIDATE:\s*(yes|no)", t, re.IGNORECASE)
    if m:
        candidate = m.group(1).lower() == "yes"
    m = re.search(r"RFI:\s*(.*)", t, re.IGNORECASE)
    if m:
        rfi = m.group(1).strip()
    return candidate, rfi


records = []
with open(LABELS_FILE) as f:
    for line in f:
        records.append(json.loads(line))

correct = total = 0
matrix = {True: {True: 0, False: 0, None: 0}, False: {True: 0, False: 0, None: 0}}
errors = []

print(f"Evaluating {len(records)} real-style images\n", flush=True)

with open(RESULTS_FILE, "w") as out:
    for i, record in enumerate(records, 1):
        filename = record["filename"]
        true_candidate = record["has_candidate"]
        response = classify(IMAGE_DIR + "/" + filename)
        pred_candidate, pred_rfi = parse(response)

        ok = pred_candidate == true_candidate
        if ok:
            correct += 1
        total += 1
        matrix[true_candidate][pred_candidate] += 1
        if not ok:
            errors.append((filename, true_candidate, pred_candidate))

        print(f"[{i}/{len(records)}] {filename} true={true_candidate} -> {pred_candidate} "
              f"({'OK' if ok else 'X'}) | RFI: {pred_rfi}", flush=True)

        out.write(json.dumps({
            "filename": filename, "has_candidate": true_candidate,
            "pred_candidate": pred_candidate, "pred_rfi": pred_rfi,
            "response": response, "correct": ok,
        }) + "\n")

print("\n" + "=" * 50)
print(f"Accuracy (real-style): {correct}/{total} = {correct / total * 100:.1f}%")
print("=" * 50)

tp = matrix[True][True]
fn = matrix[True][False]
fp = matrix[False][True]
tn = matrix[False][False]
if tp + fn > 0:
    print(f"  Recall: {tp}/{tp + fn} ({tp / (tp + fn) * 100:.0f}%)")
if tp + fp > 0:
    print(f"  Precision: {tp}/{tp + fp} ({tp / (tp + fp) * 100:.0f}%)")
print(f"  False positives: {fp}/{fp + tn}")

print("\nConfusion matrix (row = true, column = predicted): yes / no / None")
for v in [True, False]:
    print(f"  {str(v):6s}: {matrix[v][True]:3d} {matrix[v][False]:3d} {matrix[v][None]:3d}")

if errors:
    print("\nErrors:")
    for name, true_v, pred_v in errors:
        print(f"  {name}: true={true_v} pred={pred_v}")
