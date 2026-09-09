import os
import random

# ============================================================
# PANCREAS CT DATASET SPLIT
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ct",
    "images"
)

LABEL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ct",
    "labels"
)

SPLIT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ct",
    "splits"
)

# Create split directory
os.makedirs(SPLIT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Find CT files
# ------------------------------------------------------------

image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith(".nii.gz")
])

# ------------------------------------------------------------
# Check matching labels
# ------------------------------------------------------------

valid_cases = []

for image_file in image_files:

    label_file = image_file

    label_path = os.path.join(LABEL_DIR, label_file)

    if os.path.exists(label_path):
        valid_cases.append(image_file)

# ------------------------------------------------------------
# Shuffle dataset
# ------------------------------------------------------------

random.seed(42)
random.shuffle(valid_cases)

total = len(valid_cases)

train_end = int(total * 0.70)
val_end = train_end + int(total * 0.15)

train_cases = valid_cases[:train_end]
val_cases = valid_cases[train_end:val_end]
test_cases = valid_cases[val_end:]

# ------------------------------------------------------------
# Save split files
# ------------------------------------------------------------

def save_split(name, cases):

    path = os.path.join(
        SPLIT_DIR,
        name + ".txt"
    )

    with open(path, "w") as f:

        for case in cases:
            f.write(case + "\n")

    print(f"{name}: {len(cases)} cases")
    print(f"Saved to: {path}")


print("=" * 60)
print("PANCREAS CT DATASET SPLIT")
print("=" * 60)

print()

print(f"Total matching cases: {total}")

print()

save_split("train", train_cases)
save_split("validation", val_cases)
save_split("test", test_cases)

print()
print("=" * 60)
print("DATASET SPLIT COMPLETED")
print("=" * 60)