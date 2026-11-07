# DFAA-MemoryNet v1.0.0

Initial public release of the DFAA-MemoryNet model-only software package.

## Included

- DFAA-MemoryNet v9.2 dual-task convolutional encoder-decoder
- Reference 25-channel Full configuration
- Without-SMM, Forcing + current RT1, and Forcing-only ablation definitions
- Predictor construction utilities
- Soil-moisture-memory input utilities
- AdamW-based training helpers with training-only normalization
- Sliding-window spatial inference
- Evaluation metrics including PR-AUC, ROC-AUC, Brier score, CSI, F1/F2, and Recovery@15%
- Generic LOYO and LOEO split utilities
- Scientific reference checks
- Clean-install GitHub Actions workflow

## Reference specification

- Patch size: 32 x 32
- Public inference stride: 16
- Full input channels: 25
- Without-SMM channels: 16
- Forcing + current RT1 channels: 6
- Forcing-only channels: 5
- Full model parameters: 1,091,570
- Without-SMM parameters: 1,089,626
- Output channels: 2

## Scope

This is a model-only release. It does not include raw or processed study data,
manuscript-specific preprocessing, the DFAA event catalogue, manuscript figures
or tables, sensitivity analyses, trained weights, prediction caches, or derived
research products.
