# Scientific verification of the released model

The public implementation was checked against the validated DFAA-MemoryNet v9.2
workflow before release.

## Architecture

The released full model uses 25 input channels and has **1,091,570 parameters**.

The Without-SMM configuration removes exactly the nine SMM channels from the
input layer, resulting in 16 input channels and **1,089,626 parameters**.

The parameter difference is therefore:

```text
1,091,570 - 1,089,626 = 1,944
1,944 = 3 × 3 × 9 × 24
```

This confirms that the ablation changes only the first convolutional kernel and
does not alter the downstream architecture.

## Predictor configurations

- Full DFAA-MemoryNet: 25 channels
- Without SMM: 16 channels
- Forcing + current RT1: 6 channels
- Forcing only: 5 channels

The two reduced forcing baselines use the exact study definitions:

```text
Forcing + current RT1 =
pe_t + pe_tplus1 + pe_tplus2 + pe_tplus3 + pe_tplus4 + dsm_t

Forcing only =
pe_t + pe_tplus1 + pe_tplus2 + pe_tplus3 + pe_tplus4
```

## Loss

The released dual-task loss preserves the v9.2 formulation:

- 0.75 × intensity-weighted fraction MAE;
- 0.25 × event loss;
- event loss = 0.70 × weighted BCE + 0.30 × Tversky;
- Tversky alpha = 0.45;
- Tversky beta = 0.55;
- intensity weight = `1 + 4 y²`.

## Training

The released training helper preserves:

- AdamW optimizer;
- weight decay = 2e-5;
- gradient clipping with clip norm = 1.0;
- training-only normalization;
- prevalence-informed output-bias initialization;
- bounded positive-class weight in [1, 3].

The manuscript's anti-collapse hyperparameter-selection callback and
study-specific held-out-year/event orchestration are not included in this
model-only repository.

## Soil-moisture retention

The released retention utility preserves the model-input definition:

```text
retention = persistence × (1 - decay_speed)
```

## Automated check

Run:

```bash
python examples/verify_reference_spec.py
```

to verify feature counts, parameter counts, and output geometry.


## Inference convention

The released `predict_map` helper uses 32 × 32 patches, stride 16, and uniform
overlap averaging. This is the convention retained for the public model-only
release because it was used in the independent reconstruction check against
saved LOYO prediction caches.

The original study notebook also contains a separate case-study/full-grid
inference path using a denser stride and Hann-weighted blending. That
study-specific implementation is not part of this model-only release. The two
paths should therefore not be treated as the same experiment-control routine.
