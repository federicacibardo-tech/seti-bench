"""Generate a labelled dataset of 6-panel ABACAD cadence spectrograms.

Each cadence is a stack of six waterfall panels produced with setigen. The ON
panels (1, 3, 5) point at the target; the OFF panels (2, 4, 6) point at blank
sky. A cadence contains a CANDIDATE when at least one thin track appears only in
the ON panels and is absent from the OFF panels, even if an RFI track (present in
all six panels) is also present.

Dataset composition (~800 images):
    - 320 pure candidates      (an ON-only track, no RFI)
    - 280 mixed candidates     (an ON-only track next to an RFI track)
    - 200 RFI-only cadences    (no candidate)

Candidate signals use SNR 13-25 (above the model detection threshold, so they are
visible) and thin widths (4-22 Hz) to resemble real narrowband technosignatures.
Each record stores whether a candidate is present and a short description of any
RFI track, used to build the training target.
"""

import json
import numpy as np
from astropy import units as u
from astropy.time import Time
import setigen as stg
import matplotlib.pyplot as plt

plt.rcParams["image.cmap"] = "viridis"
np.random.seed(6006)

FCHANS = 256
TCHANS = 16
DF = 2.7939677238464355          # Hz per channel
DT = 18.253611008                # s per time bin
OBS_LENGTH = 300
SLEW_TIME = 15

N_PURE = 320
N_MIXED = 280
N_RFI_ONLY = 200

labels = []


def scalar(x):
    """Return a plain float from an astropy Quantity or a number."""
    return float(x.value) if hasattr(x, "value") else float(x)


def thin_width():
    """Sample a thin signal width (Hz), biased towards very narrow lines."""
    r = np.random.uniform()
    if r < 0.5:
        return np.random.uniform(4, 9)
    elif r < 0.85:
        return np.random.uniform(9, 15)
    return np.random.uniform(15, 22)


def new_cadence():
    """Build an ABACAD cadence of six noise frames."""
    fch1 = np.random.uniform(1000, 8000)
    t_start = [Time(60000, format="mjd").unix]
    for i in range(1, 6):
        t_start.append(t_start[i - 1] + OBS_LENGTH + SLEW_TIME)
    frames = []
    for i in range(6):
        frame = stg.Frame(
            fchans=FCHANS, tchans=TCHANS, df=DF * u.Hz, dt=DT * u.s,
            fch1=fch1 * u.MHz, t_start=t_start[i],
        )
        frame.add_noise(x_mean=10, noise_type="chi2")
        frames.append(frame)
    return stg.OrderedCadence(frames, order="ABACAD")


def add_candidate(cadence, index, two_on_only=False):
    """Inject an ON-only candidate track (thin, SNR 13-25).

    If two_on_only is True the signal is placed in only two of the three ON
    panels, reproducing real cadences where a candidate is not visible in every
    ON observation.
    """
    snr = np.random.uniform(13, 25)
    drift = np.random.uniform(0.03, 0.15) * np.random.choice([-1, 1])
    width = thin_width()

    on_frames = [0, 2, 4]
    if two_on_only:
        on_frames = [int(x) for x in np.random.choice([0, 2, 4], size=2, replace=False)]

    f_start = cadence[0].get_frequency(index=index)
    for idx in on_frames:
        frame = cadence[idx]
        frame.add_signal(
            stg.constant_path(f_start=f_start, drift_rate=drift * u.Hz / u.s),
            stg.constant_t_profile(level=frame.get_intensity(snr=snr)),
            stg.gaussian_f_profile(width=width * u.Hz),
            stg.constant_bp_profile(level=1),
        )
    return {"type": "candidate", "snr": snr, "width_hz": width,
            "index": int(index), "on_panels": [i + 1 for i in on_frames]}


def add_rfi_linear(cadence, index):
    snr = np.random.uniform(15, 90)
    width = thin_width()
    drift = np.random.uniform(-0.05, 0.05)
    cadence.add_signal(
        stg.constant_path(f_start=cadence[0].get_frequency(index=index),
                          drift_rate=drift * u.Hz / u.s),
        stg.constant_t_profile(level=cadence[0].get_intensity(snr=snr)),
        stg.gaussian_f_profile(width=width * u.Hz),
        stg.constant_bp_profile(level=1),
    )
    return {"type": "linear", "snr": snr, "width_hz": width, "index": int(index),
            "descr": "a straight vertical line present in all six panels"}


def add_rfi_sinusoidal(cadence, index):
    snr = np.random.uniform(15, 90)
    width = thin_width()
    period = np.random.uniform(60, 140)
    amplitude = np.random.uniform(20, 60)
    drift = np.random.uniform(-0.02, 0.02)
    f_start = scalar(cadence[0].get_frequency(index=index))
    for frame in cadence:
        t = np.arange(frame.tchans) * scalar(frame.dt)
        freqs = f_start + amplitude * np.sin(2 * np.pi * t / period) + drift * t
        frame.add_signal(
            freqs,
            stg.constant_t_profile(level=frame.get_intensity(snr=snr)),
            stg.gaussian_f_profile(width=width * u.Hz),
            stg.constant_bp_profile(level=1),
        )
    return {"type": "sinusoidal", "snr": snr, "width_hz": width, "index": int(index),
            "descr": "a wavy/oscillating line present in all six panels"}


def add_rfi_low_snr(cadence, index):
    snr = np.random.uniform(8, 15)
    width = thin_width()
    drift = np.random.uniform(-0.05, 0.05)
    cadence.add_signal(
        stg.constant_path(f_start=cadence[0].get_frequency(index=index),
                          drift_rate=drift * u.Hz / u.s),
        stg.constant_t_profile(level=cadence[0].get_intensity(snr=snr)),
        stg.gaussian_f_profile(width=width * u.Hz),
        stg.constant_bp_profile(level=1),
    )
    return {"type": "low_snr", "snr": snr, "width_hz": width, "index": int(index),
            "descr": "a faint straight line present in all six panels"}


RFI = {"linear": add_rfi_linear, "sinusoidal": add_rfi_sinusoidal, "low_snr": add_rfi_low_snr}


def separated_indices(n, min_gap=50):
    """Return n channel indices separated by at least min_gap channels."""
    chosen = []
    attempts = 0
    while len(chosen) < n and attempts < 2000:
        c = np.random.randint(30, 226)
        if all(abs(c - x) >= min_gap for x in chosen):
            chosen.append(c)
        attempts += 1
    return chosen


def make_pure_candidate():
    cadence = new_cadence()
    two_on = np.random.uniform() < 0.3
    signal = add_candidate(cadence, np.random.randint(90, 166), two_on_only=two_on)
    return cadence, {"has_candidate": True, "rfi_descr": None, "signals": [signal]}


def make_mixed_candidate():
    cadence = new_cadence()
    i_cand, i_rfi = separated_indices(2, 50)
    two_on = np.random.uniform() < 0.3
    cand = add_candidate(cadence, i_cand, two_on_only=two_on)
    rfi_type = np.random.choice(["linear", "sinusoidal", "low_snr"])
    rfi = RFI[rfi_type](cadence, i_rfi)
    return cadence, {"has_candidate": True, "rfi_descr": rfi["descr"], "signals": [cand, rfi]}


def make_rfi_only():
    cadence = new_cadence()
    rfi_type = np.random.choice(["linear", "sinusoidal", "low_snr"])
    rfi = RFI[rfi_type](cadence, np.random.randint(50, 200))
    return cadence, {"has_candidate": False, "rfi_descr": rfi["descr"], "signals": [rfi]}


def generate(group, builder, count):
    for i in range(count):
        cadence, extra = builder()
        filename = f"{group}_{i:03d}.png"
        plt.figure(figsize=(10, 10))
        cadence.plot()
        plt.savefig(filename, dpi=120)
        plt.close()
        record = {"filename": filename, "fch1_mhz": float(cadence[0].fch1) / 1e6}
        record.update(extra)
        labels.append(record)
        print(filename, "candidate" if extra["has_candidate"] else "no-candidate", flush=True)


if __name__ == "__main__":
    generate("pure_candidate", make_pure_candidate, N_PURE)
    generate("mixed_candidate", make_mixed_candidate, N_MIXED)
    generate("rfi_only", make_rfi_only, N_RFI_ONLY)

    with open("labels.jsonl", "w") as f:
        for record in labels:
            f.write(json.dumps(record) + "\n")

    n_cand = sum(1 for r in labels if r["has_candidate"])
    print(f"\nDone: {len(labels)} images "
          f"({n_cand} with candidate, {len(labels) - n_cand} without)")
