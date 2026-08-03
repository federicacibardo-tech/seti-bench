"""Generate simulated cadences rendered in the style of real observations.

The signal content is identical to the main training set (thin candidates at
SNR 13-25, some with an adjacent RFI track), but the plot is rendered to match
the appearance of real Breakthrough Listen cadences: a title line, per-panel
"Obs N (ON/OFF)" labels, a shared "Power" colorbar, a relative-frequency axis in
Hz, and wide horizontal panels.

This set is a domain-gap probe. A model trained on the default plot style can be
evaluated here to separate two failure modes: a drop in accuracy on these images
points to a plot-style domain shift, while good accuracy here isolates the real
failure to signal content or real noise rather than rendering.
"""

import json
import numpy as np
from astropy import units as u
from astropy.time import Time
import setigen as stg
import matplotlib.pyplot as plt

np.random.seed(9100)

FCHANS = 256
TCHANS = 16
DF = 2.7939677238464355
DT = 18.253611008
OBS_LENGTH = 300
SLEW_TIME = 15

N_PURE = 8
N_MIXED = 8
N_RFI_ONLY = 4

PANEL_LABELS = ["Obs 1 (ON) [ON #1]", "Obs 2 (OFF)", "Obs 3 (ON) [ON #2]",
                "Obs 4 (OFF)", "Obs 5 (ON) [ON #3]", "Obs 6 (OFF)"]

labels = []


def scalar(x):
    return float(x.value) if hasattr(x, "value") else float(x)


def thin_width():
    r = np.random.uniform()
    if r < 0.5:
        return np.random.uniform(4, 9)
    elif r < 0.85:
        return np.random.uniform(9, 15)
    return np.random.uniform(15, 22)


def new_cadence():
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
    return stg.OrderedCadence(frames, order="ABACAD"), fch1


def add_candidate(cadence, index, two_on_only=False):
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
    return {"snr": snr, "width_hz": width}


def add_rfi(cadence, index, rfi_type):
    width = thin_width()
    if rfi_type == "sinusoidal":
        snr = np.random.uniform(15, 90)
        period = np.random.uniform(60, 140)
        amplitude = np.random.uniform(20, 60)
        f_start = scalar(cadence[0].get_frequency(index=index))
        for frame in cadence:
            t = np.arange(frame.tchans) * scalar(frame.dt)
            freqs = f_start + amplitude * np.sin(2 * np.pi * t / period)
            frame.add_signal(
                freqs,
                stg.constant_t_profile(level=frame.get_intensity(snr=snr)),
                stg.gaussian_f_profile(width=width * u.Hz),
                stg.constant_bp_profile(level=1),
            )
        return "a wavy/oscillating line present in all six panels"
    snr = np.random.uniform(15, 90) if rfi_type == "linear" else np.random.uniform(8, 15)
    drift = np.random.uniform(-0.05, 0.05)
    cadence.add_signal(
        stg.constant_path(f_start=cadence[0].get_frequency(index=index),
                          drift_rate=drift * u.Hz / u.s),
        stg.constant_t_profile(level=cadence[0].get_intensity(snr=snr)),
        stg.gaussian_f_profile(width=width * u.Hz),
        stg.constant_bp_profile(level=1),
    )
    prefix = "faint " if rfi_type == "low_snr" else ""
    return f"a {prefix}straight line present in all six panels"


def separated_indices(n, min_gap=50):
    chosen = []
    while len(chosen) < n:
        c = np.random.randint(30, 226)
        if all(abs(c - x) >= min_gap for x in chosen):
            chosen.append(c)
    return chosen


def plot_real_style(cadence, fch1, filename, tic):
    """Render the cadence to match the appearance of real observations."""
    freq_hz = (np.arange(FCHANS) - FCHANS / 2) * DF
    time_s = np.arange(TCHANS) * DT

    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f"{tic} @ {fch1:.6f} MHz", fontsize=15, fontweight="bold", y=0.995)

    images = []
    for i, (ax, frame) in enumerate(zip(axes, cadence)):
        data = frame.get_data()
        norm = (data - data.min()) / (data.max() - data.min() + 1e-9)
        im = ax.imshow(norm, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                       extent=[freq_hz[0], freq_hz[-1], time_s[-1], time_s[0]])
        images.append(im)
        ax.text(0.01, 0.92, PANEL_LABELS[i], transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_ylabel("Time [s]", fontsize=10)

    axes[-1].set_xlabel(f"Relative Frequency [Hz] from {fch1:.6f} MHz", fontsize=12)

    fig.subplots_adjust(right=0.88, hspace=0.15, top=0.96)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(images[0], cax=cbar_ax).set_label("Power", fontsize=11)

    plt.savefig(filename, dpi=100, bbox_inches="tight")
    plt.close()


counter = 0


def generate(group, count, has_candidate, with_rfi):
    global counter
    for _ in range(count):
        cadence, fch1 = new_cadence()
        rfi_descr = None
        if has_candidate and with_rfi:
            i_cand, i_rfi = separated_indices(2, 50)
            add_candidate(cadence, i_cand, two_on_only=(np.random.uniform() < 0.3))
            rfi_descr = add_rfi(cadence, i_rfi,
                                np.random.choice(["linear", "sinusoidal", "low_snr"]))
        elif has_candidate:
            add_candidate(cadence, np.random.randint(90, 166),
                          two_on_only=(np.random.uniform() < 0.3))
        else:
            rfi_descr = add_rfi(cadence, np.random.randint(50, 200),
                                np.random.choice(["linear", "sinusoidal", "low_snr"]))

        tic = f"TIC{np.random.randint(10000000, 999999999)}"
        filename = f"realstyle_{counter:03d}.png"
        plot_real_style(cadence, fch1, filename, tic)
        labels.append({"filename": filename, "has_candidate": has_candidate,
                       "rfi_descr": rfi_descr, "group": group})
        print(filename, group, "candidate" if has_candidate else "no-candidate", flush=True)
        counter += 1


if __name__ == "__main__":
    generate("pure_candidate", N_PURE, True, False)
    generate("mixed_candidate", N_MIXED, True, True)
    generate("rfi_only", N_RFI_ONLY, False, True)

    with open("labels_realstyle.jsonl", "w") as f:
        for record in labels:
            f.write(json.dumps(record) + "\n")

    n_cand = sum(1 for r in labels if r["has_candidate"])
    print(f"\nDone: {len(labels)} real-style images ({n_cand} with candidate)")
