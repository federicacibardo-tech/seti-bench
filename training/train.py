"""LoRA fine-tuning of Qwen3.5-9B on the cadence dataset.

The model is trained to output a two-line answer: whether a candidate is present
and a short description of any RFI track. RFI-only samples are the minority class
for the yes/no target, so they are augmented (horizontal flip and a small
contrast change) and duplicated to balance the objective.

Run inside the NGC container. Required environment variables (set below) disable
Triton/Inductor compilation, which does not build for the sm_121 architecture
outside the container.
"""

import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import random

import torch
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True

from PIL import Image, ImageEnhance
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

IMAGE_DIR = "/workspace"
TRAIN_FILE = "train.jsonl"
BASE_MODEL = "Qwen/Qwen3.5-9B"
OUTPUT_DIR = "qwen35_finetuned"

EPOCHS = 2
LEARNING_RATE = 2e-5
BATCH_SIZE = 1
GRAD_ACCUM = 4

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


def load(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def build_target(record):
    candidate = "yes" if record["has_candidate"] else "no"
    rfi = record.get("rfi_descr") or "none"
    return f"CANDIDATE: {candidate}\nRFI: {rfi}"


def augment(img):
    """Light augmentation that preserves the ON/OFF panel order."""
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    factor = random.uniform(0.9, 1.1)
    img = ImageEnhance.Contrast(img).enhance(factor)
    return img


def to_conversation(record, do_augment=False):
    img = Image.open(IMAGE_DIR + "/" + record["filename"]).convert("RGB")
    if do_augment:
        img = augment(img)
    return {"messages": [
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": PROMPT},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": build_target(record)}]},
    ]}


def main():
    print("Loading base model...")
    model, processor = FastVisionModel.from_pretrained(
        BASE_MODEL, load_in_4bit=True, use_gradient_checkpointing="unsloth",
    )

    print("Adding LoRA adapters...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=42,
    )

    print("Preparing data...")
    records = load(TRAIN_FILE)
    with_candidate = [r for r in records if r["has_candidate"]]
    without_candidate = [r for r in records if not r["has_candidate"]]

    dataset = [to_conversation(r) for r in with_candidate]
    for r in without_candidate:
        dataset.append(to_conversation(r))
        dataset.append(to_conversation(r, do_augment=True))   # augmented duplicate

    random.seed(42)
    random.shuffle(dataset)
    print(f"Training examples: {len(dataset)}")

    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=SFTConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            warmup_steps=20,
            lr_scheduler_type="cosine",
            logging_steps=5,
            output_dir=OUTPUT_DIR,
            save_strategy="steps",
            save_steps=50,
            optim="adamw_8bit",
            weight_decay=0.01,
            seed=42,
            dataloader_num_workers=0,
            report_to="none",
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=4096,
        ),
    )

    print("\nTraining...\n")
    trainer.train()

    print("\nSaving...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
