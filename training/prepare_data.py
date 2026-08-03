"""Stratified train/test split of the cadence dataset.

The split is stratified over three strata (pure candidate, mixed candidate,
RFI-only) so that the test set contains a representative share of each. Runs on
CPU; no GPU required.
"""

import json
import random
from collections import Counter

LABELS_FILE = "labels.jsonl"
TEST_FRACTION = 0.12

records = []
with open(LABELS_FILE) as f:
    for line in f:
        records.append(json.loads(line))


def stratum(record):
    if record["has_candidate"]:
        return "mixed" if record.get("rfi_descr") else "pure"
    return "rfi_only"


groups = {}
for record in records:
    groups.setdefault(stratum(record), []).append(record)

random.seed(42)
train, test = [], []
for stratum_records in groups.values():
    random.shuffle(stratum_records)
    n_test = max(1, int(len(stratum_records) * TEST_FRACTION))
    test.extend(stratum_records[:n_test])
    train.extend(stratum_records[n_test:])

random.shuffle(train)
random.shuffle(test)

with open("train.jsonl", "w") as f:
    for record in train:
        f.write(json.dumps(record) + "\n")
with open("test.jsonl", "w") as f:
    for record in test:
        f.write(json.dumps(record) + "\n")

print(f"Train: {len(train)}  Test: {len(test)}")


def summary(name, subset):
    with_candidate = sum(1 for r in subset if r["has_candidate"])
    print(f"{name}: with_candidate={with_candidate}  without={len(subset) - with_candidate}")
    print(f"   strata={dict(Counter(stratum(r) for r in subset))}")


summary("Train", train)
summary("Test", test)
