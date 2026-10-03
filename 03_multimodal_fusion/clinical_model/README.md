# Clinical Model

This module implements the clinical-only baseline for the multimodal pancreatic cancer detection system.

## Purpose

The clinical model converts patient-level clinical information into a numerical feature representation and predicts the probability of pancreatic ductal adenocarcinoma (PDAC).

The clinical pipeline is intentionally developed separately from CT-based modeling before multimodal fusion.

## Input Data

The model uses the processed clinical datasets located in:

```text
data/processed/clinical/