# DFAA-MemoryNet input contract

The model-only repository assumes that the study-specific preprocessing stage
has already produced spatially aligned weekly predictor fields.

## Required dimensions

Dynamic fields:

```text
time × lat × lon
```

Static fields:

```text
lat × lon
```

Prescribed future forcing:

```text
lead × time × lat × lon
```

with `lead = 1, 2, 3, 4`.

## Full reference feature order

The exact 25-channel order is defined by
`dfaa_memorynet.predictors.FULL_FEATURE_NAMES`.

## Target tensor

Training targets use:

```text
[..., 0] = four-week hotspot fraction
[..., 1] = binary event/core target
[..., 2] = valid-pixel mask
```

## Data not distributed

This repository intentionally does not define how the manuscript's raw
Earth-observation products are transformed into these fields. That processing
belongs to the study data workflow rather than to the DFAA-MemoryNet software.
