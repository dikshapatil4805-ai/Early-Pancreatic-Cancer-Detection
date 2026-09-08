import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from data_loader import CTClassificationDataset


print("=" * 60)
print("CT DATASET CLASS DISTRIBUTION")
print("=" * 60)

for split in ["train", "validation", "test"]:

    dataset = CTClassificationDataset(split=split)

    class_0 = 0
    class_1 = 0

    for i in range(len(dataset)):

        _, target, _ = dataset[i]

        target = target.item()

        if target == 0:
            class_0 += 1
        else:
            class_1 += 1

    print()
    print(f"{split.upper()}")
    print("-" * 30)
    print(f"Class 0 : {class_0}")
    print(f"Class 1 : {class_1}")
    print(f"Total   : {class_0 + class_1}")

print()
print("=" * 60)