"""Generate a controlled SNR-detectability test set.

The candidate signal width is held fixed (thin, 3 Hz) so that SNR is the only
variable. Candidates are produced at ten fixed SNR levels (7 to 25), ten per
level. A set of pure-noise cadences is added to measure the false-positive rate:
a model that simply answers "candidate" everywhere will score high recovery but
will also flag noise, which this control exposes.

Classes: candidate vs noise_only.
"""

import json
import numpy as np
from astropy import units as u
from astropy.time import Time
import setigen as stg
import matplotlib.pyplot as plt

plt.rcParams["image.cmap"] = "viridis"
np.random.seed(7000)

FCHANS = 256
TCHANS = 16
DF = 2.7939677238464355
DT = 18.253611008
OBS_LENGTH = 300
SLEW_TIME = 15

SNR_LEVELS = [7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
PER_LEVEL = 10
FIXED_WIDTH = 3.0        # Hz, thin and constant
N_NOISE = 30             # pure-noise cadences for false-positive control

labels = []


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
    return stg.OrderedCadence(frames, order="ABACAD")


def candidate_at_snr(snr):
    cadence = new_cadence()
    drift = np.random.uniform(0.03, 0.15) * np.random.choice([-1, 1])
    index = np.random.randint(100, 156)
    on = cadence.by_label("A")
    on.add_signal(
        stg.constant_path(f_start=cadence[0].get_frequency(index=index),
                          drift_rate=drift * u.Hz / u.s),
        stg.constant_t_profile(level=cadence[0].get_intensity(snr=snr)),
        stg.gaussian_f_profile(width=FIXED_WIDTH * u.Hz),
        stg.constant_bp_profile(level=1),
    )
    return cadence, {"snr": float(snr), "drift_rate_hz_s": drift,
                     "width_hz": FIXED_WIDTH, "index": int(index)}


if __name__ == "__main__":
    counter = 0
    for snr in SNR_LEVELS:
        for _ in range(PER_LEVEL):
            cadence, params = candidate_at_snr(snr)
            filename = f"snrtest_{counter:03d}.png"
            plt.figure(figsize=(10, 10))
            cadence.plot()
            plt.savefig(filename, dpi=120)
            plt.close()
            labels.append({"filename": filename, "class": "candidate",
                           "fch1_mhz": float(cadence[0].fch1) / 1e6, "params": params})
            print(f"[candidate {counter + 1}/100] {filename} SNR={snr}", flush=True)
            counter += 1

    for j in range(N_NOISE):
        cadence = new_cadence()          # noise only, no signal injected
        filename = f"noisetest_{j:03d}.png"
        plt.figure(figsize=(10, 10))
        cadence.plot()
        plt.savefig(filename, dpi=120)
        plt.close()
        labels.append({"filename": filename, "class": "noise_only",
                       "fch1_mhz": float(cadence[0].fch1) / 1e6,
                       "params": {"snr": None, "width_hz": None}})
        print(f"[noise {j + 1}/{N_NOISE}] {filename}", flush=True)

    with open("labels_snrtest.jsonl", "w") as f:
        for record in labels:
            f.write(json.dumps(record) + "\n")

    print(f"\nDone: 100 candidates (SNR {min(SNR_LEVELS)}-{max(SNR_LEVELS)}, "
          f"width {FIXED_WIDTH} Hz) + {N_NOISE} noise-only")
