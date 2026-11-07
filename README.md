# DFAA-MemoryNet

**DFAA-MemoryNet** is a dual-task convolutional encoder–decoder developed to model
the spatial response of near-surface soil moisture to drought–flood abrupt
alternation (DFAA) under prescribed hydroclimatic forcing.

The model combines current hydroclimatic and soil-moisture conditions,
antecedent soil-moisture-memory (SMM) descriptors, prescribed future PEAI
forcing, and static spatial predictors to estimate the spatial organization of
DFAA soil-moisture response hotspots over a four-week horizon.

> **Scope of this repository:** this is a **model-only software repository**.
> It provides the DFAA-MemoryNet implementation and reusable model utilities.
> The study-specific preprocessing workflow, datasets, manuscript figures,
> sensitivity analyses, event catalogue, and derived research products are not
> distributed here.

## Included

- DFAA-MemoryNet v9.2 architecture;
- dual-task intensity-aware loss;
- predictor-construction utilities;
- utilities for preparing SMM predictor inputs;
- training and normalization helpers;
- sliding-window spatial inference;
- generic LOYO and LOEO split utilities;
- PR-AUC, ROC-AUC, Brier score, CSI, F1/F2, and equal-area Recovery@15%;
- reference model configuration;
- synthetic quick-start example.

## Not included

- raw or processed Sentinel-1 / PEAI datasets;
- basin-specific preprocessing scripts;
- manuscript-specific DTF/FTD event detection and event catalogue;
- manuscript figures and tables;
- threshold-sensitivity experiments;
- event-by-event scientific analyses;
- prediction caches;
- trained model weights;
- derived research products.

## Reference model inputs

The full DFAA-MemoryNet configuration uses 25 predictor channels:

1. `pe_t`
2. `dsm_t`
3. `pe_mean2`
4. `pe_mean4`
5. `dsm_mean2`
6. `dsm_mean4`
7. `pe_slope1`
8. `dsm_slope1`
9. `memory_persistence_t`
10. `memory_decay_speed_t`
11. `memory_instability_t`
12. `memory_retention_t`
13. `pe_tplus1`
14. `pe_tplus2`
15. `pe_tplus3`
16. `pe_tplus4`
17. `persistence_static`
18. `decay_speed_static`
19. `instability_static`
20. `memory_retention_static`
21. `memory_class`
22. `land_organization`
23. `x_norm`
24. `y_norm`
25. `valid_mask`

The future PEAI channels (`pe_tplus1` to `pe_tplus4`) represent **prescribed
future forcing**. Future observed RT1 soil-moisture response is not used as a
predictor.

## Predictor configurations

The package supports the four configurations used to evaluate the contribution
of soil-moisture memory:

- **Full DFAA-MemoryNet**
- **Without SMM**
- **Forcing + current RT1**
- **Forcing only**

The exact feature lists are defined in
`src/dfaa_memorynet/predictors.py`.

## Model outputs

DFAA-MemoryNet returns two pixel-wise output channels:

- **Channel 0:** predicted four-week hotspot fraction;
- **Channel 1:** hotspot-event probability.

During training, the target tensor contains:

- four-week hotspot fraction;
- binary event/core target;
- valid-pixel mask.

## Reference model geometry

- patch size: **32 × 32 pixels**;
- overlap inference stride: **16 pixels**;
- response horizon: **4 weeks**;
- full configuration: **25 predictor channels**.

## Installation

```bash
git clone https://github.com/imane-se/DFAA-MemoryNet.git
cd DFAA-MemoryNet
pip install -e .
```

## Quick test

A synthetic smoke test is provided to verify that the package and model
architecture are correctly installed:

```bash
python examples/quickstart.py
```

The quick test does not reproduce manuscript results and does not require study
data.

## Basic use

```python
from dfaa_memorynet import build_memorynet, FULL_FEATURE_NAMES

model = build_memorynet(
    input_shape=(32, 32, len(FULL_FEATURE_NAMES)),
    base_filters=24,
    dropout_scale=1.05,
)

model.summary()
```

## Validation utilities

`validation.py` provides reusable generators for:

- leave-one-year-out (**LOYO**) splits;
- leave-one-event-out (**LOEO**) splits.

The construction of the manuscript-specific DFAA event catalogue, event
buffers, and study-specific exclusion rules is intentionally outside the scope
of this repository.

## Reproducibility scope

This repository provides the **software implementation of DFAA-MemoryNet** and
the utilities required to construct, train, evaluate, and apply the model once
the required predictor fields have been prepared.

It is not intended to reproduce every figure, table, sensitivity experiment, or
derived research product reported in the associated manuscript.

## Citation

Citation information for the software is provided in `CITATION.cff`. The
associated article citation and archived software DOI will be added to the
versioned release.

## License

This software is distributed under the MIT License.
