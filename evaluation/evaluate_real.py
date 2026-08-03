"""Run the fine-tuned model on a folder of real cadence images.

Real images are larger than the simulated ones, so they are downscaled with
thumbnail() to keep the image-token count aligned with the text template. No
ground truth is assumed; predictions are printed and saved for manual review.
Run inside the NGC container.
"""

import os
import json
import re
from glob import glob

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from PIL import Image
from unsloth import FastVisionModel

torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True

REAL_DIR = "/workspace/real_cadences"
MODEL_DIR = "qwen35_finetuned"
RESULTS_FILE = "results_real.jsonl"

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
    img.thumbnail((1400, 1400))          # keep image-token count aligned
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


extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG")
images = []
for ext in extensions:
    images.extend(glob(os.path.join(REAL_DIR, ext)))
images.sort()

if not images:
    print(f"No images found in: {REAL_DIR}")
    raise SystemExit(1)

print(f"Found {len(images)} real images. Running inference...\n", flush=True)

candidates_found = 0
rfi_found = 0

with open(RESULTS_FILE, "w") as out:
    for i, path in enumerate(images, 1):
        name = os.path.basename(path)
        response = classify(path)
        pred_candidate, pred_rfi = parse(response)

        if pred_candidate is True:
            candidates_found += 1
        if pred_rfi and pred_rfi.lower() != "none":
            rfi_found += 1

        print(f"[{i}/{len(images)}] {name}", flush=True)
        print(f"   CANDIDATE: {'yes' if pred_candidate else 'no'} | RFI: {pred_rfi}", flush=True)

        out.write(json.dumps({
            "filename": name, "pred_candidate": pred_candidate,
            "pred_rfi": pred_rfi, "response": response,
        }) + "\n")

print("\n" + "=" * 50)
print(f"Summary ({len(images)} real images):")
print(f"  Candidates predicted (yes): {candidates_found}")
print(f"  Images with RFI described: {rfi_found}")
print(f"  Results saved to: {RESULTS_FILE}")
print("=" * 50)
