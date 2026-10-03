# Multimodal Fusion

This folder contains the multimodal fusion architectures for combining CT-derived
features with clinical features for pancreatic ductal adenocarcinoma (PDAC) risk
prediction.

## Current architecture

The multimodal pipeline is designed as:

CT image
→ CT encoder
→ 512-dimensional CT feature vector

Clinical data
→ clinical preprocessing
→ Clinical MLP
→ 16-dimensional clinical feature vector

CT features + Clinical features
→ Multimodal fusion
→ Binary PDAC risk prediction

## Implemented fusion methods

### 1. Concatenation Fusion

File:

`concatenation.py`

The CT and clinical feature vectors are concatenated:

```text
CT features       : 512 dimensions
Clinical features : 16 dimensions
                         ↓
Concatenation
                         ↓
Fused representation: 528 dimensions
                         ↓
Linear(528 → 128)
                         ↓
ReLU + Dropout
                         ↓
Linear(128 → 1)
                         ↓
PDAC risk logit