# SETI Bench

Benchmarking small vision-language models (VLMs) for classifying radio
technosignature candidates in 6-panel ABACAD cadence spectrograms.

Each image is a stack of six waterfall panels. In an ABACAD cadence the telescope
alternates between the target star (**ON** — panels 1, 3, 5) and blank sky
(**OFF** — panels 2, 4, 6). A real sky source appears only in the ON panels;
terrestrial interference (RFI) appears in all six. The task is to decide whether a
cadence contains a **candidate** (a narrowband track present only in the ON
panels, even alongside RFI) and to describe any RFI present.

---

## Key results

| Approach | Test set | Result |
|---|---|---|
| Qwen3.5-9B, few-shot (3 examples), viridis | 297 simulated cadences | **100%** |
| Qwen3.5-9B, fine-tuned (LoRA) | held-out simulated test | **94.7%** (precision 100%, 0 false positives, recall 93%) — in-distribution only, see *Next steps* |
| Qwen3.5-9B, few-shot, multi-track prompt | real Breakthrough Listen cadences | 8–9 / 9 clear candidates |

Three findings:

1. **The colormap benefit is model-dependent.** A high-contrast colormap adds ~40
   points to a weaker model (Qwen2.5-VL: 47.7% → 87.7%) but nothing to a stronger
   one (Qwen3.5-9B: 100% in viridis). Working in viridis matches the real-data
   colormap with no accuracy loss.


2. **Detection must be measured with a false-positive control.** In a fixed-width
   SNR-detectability test, four of seven models reach 100% recovery simply by
   answering "candidate" everywhere — they also flag 97–100% of pure-noise images.
   Only three models are genuine detectors: Qwen3.5-9B (threshold ~SNR 13, 0% false
   positives), Qwen3.5-2B (~SNR 11, 10%), and Gemma 4B (~SNR 17, 0%).

3. **Fine-tuning is promising but not yet production-ready.** The fine-tuned model
performs strongly in-distribution (94.7% on the simulated test, precision 100%,
zero false positives), which shows the approach is viable. On real images,
however, it does not yet generalise: the rendering of real cadences differs from
the training set, and the model tends to fall back on the most frequent training
answer rather than inspecting each image. Closing this gap is the main open work
item — see *Next steps*. For now, few-shot, which reasons per image rather than
specialising on one plot style, is the more reliable route on real data.

---

## Repository layout

```
data_generation/
    generate_cadences.py        Main labelled dataset (~800 cadences)
    generate_snr_test.py        Fixed-width SNR-detectability set + pure-noise control
    generate_real_style.py      Simulated cadences rendered in real-observation style
training/
    prepare_data.py             Stratified train/test split
    train.py                    LoRA fine-tuning of Qwen3.5-9B
evaluation/
    evaluate_finetuned.py       Fine-tuned model on the simulated test set
    evaluate_real.py            Fine-tuned model on real cadences
    evaluate_real_style.py      Fine-tuned model on real-style images (domain-gap probe)
    evaluate_snr_detectability.py   Recovery + false-positive rate vs SNR (Ollama)
fewshot/
    fewshot_simulated.py        Few-shot classification of simulated cadences
    fewshot_real.py             Few-shot classification of real cadences
figures/
    make_figures.py             Reproduce the analysis figures
    *.png                       Generated figures
```

---

## Data model

Cadences are generated with [setigen](https://github.com/bbrzycki/setigen).
Fixed parameters: 256 frequency channels, 16 time bins, channel width
2.79 Hz, time bin 18.25 s, ABACAD ordering, viridis colormap.

Signal types:

- **Candidate** — a thin ON-only track (SNR 13–25, width 4–22 Hz). Some appear in
  only two of the three ON panels, as in real cadences.
- **RFI** — linear, sinusoidal, or low-SNR tracks present in all six panels.

Labels are written to `labels.jsonl`, one JSON object per image, recording
`has_candidate`, an RFI description string, and signal parameters.

---

## Reproducing the pipeline

### 1. Generate data (CPU)

```bash
python data_generation/generate_cadences.py       # -> images + labels.jsonl
python data_generation/generate_snr_test.py        # -> SNR test + labels_snrtest.jsonl
python data_generation/generate_real_style.py      # -> real-style set + labels_realstyle.jsonl
```

### 2. Few-shot baseline (Ollama)

```bash
python fewshot/fewshot_simulated.py                # simulated cadences
python fewshot/fewshot_real.py                     # real cadences (folder: real_cadences/)
```

### 3. Fine-tuning (GPU container)

```bash
python training/prepare_data.py                    # -> train.jsonl / test.jsonl
python training/train.py                           # -> qwen35_finetuned/
python evaluation/evaluate_finetuned.py            # simulated test set
python evaluation/evaluate_real.py                 # real cadences
python evaluation/evaluate_real_style.py           # real-style domain-gap probe
```

### 4. SNR-detectability study (Ollama)

Set `MODEL` in the script and run once per model:

```bash
python evaluation/evaluate_snr_detectability.py
```

### 5. Figures

```bash
cd figures && python make_figures.py
```

---

## Environment

**Few-shot / detectability** run through [Ollama](https://ollama.com). Qwen3.5
requires `think=False` in the chat call for usable latency.

**Fine-tuning** uses [Unsloth](https://github.com/unslothai/unsloth) with 4-bit
LoRA, run inside the NGC PyTorch container
(`nvcr.io/nvidia/pytorch:25.10-py3`) on an NVIDIA GB10 (Grace Blackwell, sm_121).
The following environment variables must be set at the top of every training and
evaluation script, because Triton/Inductor kernels do not compile for sm_121
outside the container:

```python
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
```

Libraries are installed per session (the container starts clean):

```bash
pip install unsloth timm trl pillow
pip install --upgrade "torchao>=0.16.0"
pip install "git+https://github.com/huggingface/transformers.git@main"
```

Real images are larger than the simulated ones; the real-data evaluation scripts
downscale them with `Image.thumbnail((1400, 1400))` to keep the image-token count
aligned with the text template.

---

## Next steps

The fine-tuned model is the main area of active work. Priorities:

- **Close the simulated-to-real gap.** Train on cadences rendered in the real
  observation style (title, colorbar, panel labels, aspect ratio) so the model
  sees the target distribution during training. The `generate_real_style.py` set
  and `evaluate_real_style.py` probe are the first step towards isolating and
  fixing this.
- **Reduce target memorisation.** Vary the RFI descriptions and reduce repetition
  in the training targets so the model learns the ON/OFF rule rather than the
  answer strings; evaluate intermediate checkpoints rather than only the final one.
- **Broaden the training signal set.** Add broadband and further signal
  morphologies, and expand the mixed-candidate cases (a thin ON-only track next to
  a bright RFI track), which are the hardest and where recall currently drops.
- **Report operational metrics.** Move from balanced accuracy to precision/recall
  and false positives per observing hour — the metric that matters for a real
  pipeline.

## Notes on method

- **Few-shot prompt.** A counting procedure (count tracks → check each against the
  OFF panels → decide) is more robust than a single holistic judgement on
  multi-track cadences. Adding further instructions past a certain point degrades
  accuracy; the counting procedure alone reaches ~93.5% on the multi-track task.

- **Fine-tuning target.** The model is trained to emit `CANDIDATE: yes/no` plus a
  short RFI description. Repetitive targets are easy to memorise; the RFI-only
  minority class is augmented to keep the yes/no objective balanced.

- **Reading the SNR curves.** Recovery alone is misleading — it must be read with
  the false-positive rate on pure noise. The 2σ error bars use the binomial
  standard error; with ten samples per bin the Gaussian approximation is loose near
  0% and 100%, where a Wilson interval would be more accurate.
