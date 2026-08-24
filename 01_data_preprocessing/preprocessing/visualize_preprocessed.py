import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

ct_path = "data/processed/ct/images/pancreas_001.nii.gz"
label_path = "data/processed/ct/labels/pancreas_001.nii.gz"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading processed CT...")
ct = nib.load(ct_path).get_fdata()

print("Loading processed label...")
label = nib.load(label_path).get_fdata()


# --------------------------------------------------
# INFORMATION
# --------------------------------------------------

print("\nPROCESSED CT")
print("----------------------------------------")
print("Shape:", ct.shape)
print("Minimum:", ct.min())
print("Maximum:", ct.max())

print("\nLABEL")
print("----------------------------------------")
print("Shape:", label.shape)
print("Unique values:", np.unique(label))


# --------------------------------------------------
# FIND A SLICE CONTAINING PANCREAS/CANCER
# --------------------------------------------------

label_voxels = np.sum(label > 0, axis=(0, 1))

slice_index = np.argmax(label_voxels)

print("\nVISUALIZATION")
print("----------------------------------------")
print("Selected slice:", slice_index)
print("Label voxels in slice:", label_voxels[slice_index])


# --------------------------------------------------
# GET SLICE
# --------------------------------------------------

ct_slice = ct[:, :, slice_index]
label_slice = label[:, :, slice_index]


# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(ct_slice.T, cmap="gray", origin="lower")
plt.title("Processed CT")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(label_slice.T, cmap="viridis", origin="lower")
plt.title("Segmentation Label")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(ct_slice.T, cmap="gray", origin="lower")
plt.imshow(
    np.ma.masked_where(label_slice.T == 0, label_slice.T),
    cmap="autumn",
    alpha=0.5,
    origin="lower"
)
plt.title("CT + Segmentation")
plt.axis("off")

plt.tight_layout()
plt.show()


print("\n============================================================")
print("VISUALIZATION COMPLETED")
print("============================================================")