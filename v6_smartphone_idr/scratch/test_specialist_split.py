"""
test_specialist_split.py
------------------------
Inspects journey scenario assignments and verifies sample distributions for Model A and Model B.
"""

import json
import re
from pathlib import Path
import numpy as np
import pypdf

import sys
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from v6_smartphone_idr.src.preprocess_v6 import (
    find_synchronized_pairs,
    process_journey_v6,
    TEST_SCENARIOS,
    WINDOW_SIZE,
    CLIP,
    DT,
)

ROOT = Path("v6_smartphone_idr")
CONFIG_PATH = ROOT / "config" / "v6_config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

# Parse PDF descriptions
pdf_path = Path("C:/Users/tiwar/OneDrive/Desktop/DATASETS/IO-VNBD/IO-VNBD/README_1.pdf")
reader = pypdf.PdfReader(str(pdf_path))
full_pdf = ""
for page in reader.pages[5:15]:
    full_pdf += page.extract_text() + "\n"

pattern = re.compile(r"V-(V[a-z0-9]+|S[0-9a-z]+|M|Y[0-9]*)", re.IGNORECASE)
journey_desc = {}
cur_j = None
cur_text = []
for line in full_pdf.split("\n"):
    m = pattern.search(line)
    if m:
        if cur_j:
            journey_desc[cur_j.lower()] = " ".join(cur_text)
        cur_j = m.group(1).lower().replace("-", "").replace("_", "")
        cur_text = [line]
    elif cur_j:
        cur_text.append(line)
if cur_j:
    journey_desc[cur_j.lower()] = " ".join(cur_text)

# Check all pairs
pairs = find_synchronized_pairs()
all_names = sorted(list(pairs.keys()))

test_named_tags = set()
for tags in TEST_SCENARIOS.values():
    for tag in tags:
        test_named_tags.add(tag.lower().replace("-", "").replace("_", ""))

test_journeys = [k for k in all_names if k in test_named_tags]
remaining = [k for k in all_names if k not in test_named_tags]

np.random.seed(42)
indices = np.random.permutation(len(remaining))
pool = [remaining[i] for i in indices]

idx = 0
while len(test_journeys) < 13:
    test_journeys.append(pool[idx])
    idx += 1

val_journeys = []
while len(val_journeys) < 9:
    val_journeys.append(pool[idx])
    idx += 1

train_journeys = pool[idx:]

print(f"Total journeys: {len(all_names)} (Train: {len(train_journeys)}, Val: {len(val_journeys)}, Test: {len(test_journeys)})")

# Classify trips into Family A and Family B
def classify_trip(name):
    desc = journey_desc.get(name.lower(), "").lower()
    is_turn = ("round" in desc and "about" in desc) or ("sharp" in desc) or ("turns" in desc) or ("turn" in desc) or (name.lower() in ["vw6", "vw7", "vw8", "vta11", "vw5"])
    is_long = ("motorway" in desc) or ("hard brake" in desc) or ("brake" in desc) or ("speed" in desc) or ("accel" in desc) or ("straight" in desc) or (name.lower() in ["vw12", "vta12", "vta9", "vw16b", "vw17", "vtb1", "vta15"])
    return is_turn, is_long

trips_A = [t for t in train_journeys if classify_trip(t)[0]]
trips_B = [t for t in train_journeys if classify_trip(t)[1]]

print(f"Train trips for Model A (Turning): {len(trips_A)} trips -> {trips_A[:10]}")
print(f"Train trips for Model B (Longitudinal): {len(trips_B)} trips -> {trips_B[:10]}")

val_A = [t for t in val_journeys if classify_trip(t)[0]]
val_B = [t for t in val_journeys if classify_trip(t)[1]]
print(f"Val trips for Model A: {len(val_A)} trips -> {val_A}")
print(f"Val trips for Model B: {len(val_B)} trips -> {val_B}")
