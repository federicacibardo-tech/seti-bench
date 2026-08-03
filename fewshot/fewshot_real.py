"""Few-shot classification of real cadences (candidate vs RFI).

Applies the multi-track counting procedure to real images. The prompt emphasises
that a candidate can be an extremely thin, faint line and that diffuse background
brightness is not a signal. No ground truth is assumed; predictions and reasoning
are printed for manual review. Runs on Ollama.
"""

import os
import glob

import ollama

REAL_DIR = "real_cadences"
EXAMPLE_DIR = "."
MODEL = "qwen3.5:9b"

CLASSES = ["candidate", "rfi"]

EXAMPLE_CANDIDATE = os.path.join(EXAMPLE_DIR, "pure_candidate_000.png")
EXAMPLE_MIXED = os.path.join(EXAMPLE_DIR, "mixed_candidate_000.png")
EXAMPLE_RFI = os.path.join(EXAMPLE_DIR, "rfi_only_000.png")

PROMPT = (
    "You are a brightness detector for a 6-panel spectrogram (1 to 6 from top to bottom).\n"
    "- Panels 1, 3, 5 are ON-target.\n"
    "- Panels 2, 4, 6 are OFF-target.\n\n"
    "A signal is a recognizable thin track against a noisy background - it can be a very "
    "thin near-vertical line, only a few pixels wide, straight, tilted, or wavy. A "
    "candidate line can be EXTREMELY THIN and FAINT, like a thread.\n\n"
    "An image can contain one or several tracks at DIFFERENT horizontal positions.\n\n"
    "Two important cases:\n"
    "- A candidate track (ON-only) can be a very thin, faint line, MUCH weaker than a "
    "bright RFI track next to it, and must be distinguished from the diffuse background "
    "noise. Look hard for a thin ON-only line even when other things are brighter.\n"
    "- An RFI track can also be faint while running through ALL panels. If a thin line "
    "appears at the same frequency in every panel, it is RFI, not a candidate.\n\n"
    "FOLLOW THIS PROCEDURE:\n"
    "1. Count how many distinct bright tracks you see (look across the whole width).\n"
    "2. For EACH track, check its horizontal position: does a track exist at that SAME "
    "position in the OFF panels (2, 4, 6)?\n"
    "   - track present in ON but NOT in OFF at that position = ON-only track\n"
    "3. Decide:\n"
    "   - CLASS 'candidate': at least ONE track is ON-only. (Even if OTHER tracks are "
    "RFI present everywhere - one ON-only track is enough to make it a candidate.)\n"
    "   - CLASS 'rfi': every track appears in all 6 panels, and NO track is ON-only.\n\n"
    "IMPORTANT: if there is only ONE track and it appears in all panels, it is rfi. "
    "Do not call it candidate just because it looks a bit brighter in some ON panels.\n\n"
    "Answer immediately.\n"
    "Format your output exactly as follows (no markdown, no code blocks):\n"
    "REASON: [how many tracks, and is any of them ON-only?]\n"
    "CLASS: [Write exactly one of: candidate or rfi]"
)

ANSWER_CANDIDATE = ("REASON: One thin track, visible only in the ON panels (1, 3, 5) and "
                    "absent from the OFF panels, so it is ON-only.\nCLASS: candidate")
ANSWER_MIXED = ("REASON: Two tracks: one appears in all panels (RFI), the other is a "
                "thin line only in the ON panels, so it is a candidate.\nCLASS: candidate")
ANSWER_RFI = ("REASON: One track, present in all 6 panels including the OFF ones; "
              "no ON-only track.\nCLASS: rfi")


def classify(path):
    messages = [
        {"role": "user", "content": PROMPT, "images": [EXAMPLE_CANDIDATE]},
        {"role": "assistant", "content": ANSWER_CANDIDATE},
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


images = sorted(glob.glob(REAL_DIR + "/*.png"))
print(f"Classifying {len(images)} real images from '{REAL_DIR}'\n", flush=True)
counts = {"candidate": 0, "rfi": 0, "UNCLEAR": 0}

for path in images:
    name = os.path.basename(path)
    response = classify(path)
    predicted = parse(response)
    counts[predicted] = counts.get(predicted, 0) + 1
    print("=" * 60, flush=True)
    print(f"{name}  ->  {predicted}", flush=True)
    print(response.strip(), flush=True)
    print(flush=True)

print("=" * 60)
print("Summary:", counts)
