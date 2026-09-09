from pathlib import Path


# Find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset folders
IMAGES_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ct"
    / "Task07_Pancreas"
    / "imagesTr"
)

LABELS_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ct"
    / "Task07_Pancreas"
    / "labelsTr"
)


def get_patient_ids(folder):
    """Get patient IDs from NIfTI files."""
    return {
    file.name.replace(".nii.gz", "")
    for file in folder.glob("*.nii.gz")
    if not file.name.startswith("._")
}


def main():

    images = get_patient_ids(IMAGES_DIR)
    labels = get_patient_ids(LABELS_DIR)

    print("=" * 50)
    print("PANCREAS CT DATASET CHECK")
    print("=" * 50)

    print(f"CT images found : {len(images)}")
    print(f"Labels found    : {len(labels)}")

    missing_labels = images - labels
    missing_images = labels - images

    print("\nCT images without labels:")

    if missing_labels:
        for item in sorted(missing_labels):
            print("   ", item)
    else:
        print("   None")

    print("\nLabels without CT images:")

    if missing_images:
        for item in sorted(missing_images):
            print("   ", item)
    else:
        print("   None")

    print("\nMatched pairs:")
    print(f"   {len(images & labels)}")


if __name__ == "__main__":
    main()