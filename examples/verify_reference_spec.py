"""Scientific reference checks for the released DFAA-MemoryNet implementation.

Run:
    python examples/verify_reference_spec.py
"""

from dfaa_memorynet import build_memorynet
from dfaa_memorynet.predictors import feature_names_for_variant


EXPECTED = {
    "full": 25,
    "without_smm": 16,
    "forcing_current_rt1": 6,
    "forcing_only": 5,
}

for variant, expected_channels in EXPECTED.items():
    names = feature_names_for_variant(variant)
    assert len(names) == expected_channels, (
        f"{variant}: expected {expected_channels} channels, got {len(names)}"
    )

full_model = build_memorynet((32, 32, EXPECTED["full"]))
without_model = build_memorynet((32, 32, EXPECTED["without_smm"]))

assert full_model.count_params() == 1_091_570, (
    f"Full parameter count mismatch: {full_model.count_params()}"
)
assert without_model.count_params() == 1_089_626, (
    f"Without-SMM parameter count mismatch: {without_model.count_params()}"
)
assert full_model.output_shape == (None, 32, 32, 2)
assert without_model.output_shape == (None, 32, 32, 2)

assert full_model.count_params() - without_model.count_params() == 1_944

print("DFAA-MemoryNet reference specification: OK")
print("Full channels:", EXPECTED["full"])
print("Without-SMM channels:", EXPECTED["without_smm"])
print("Forcing + current RT1 channels:", EXPECTED["forcing_current_rt1"])
print("Forcing-only channels:", EXPECTED["forcing_only"])
print("Full parameters:", full_model.count_params())
print("Without-SMM parameters:", without_model.count_params())
print("Parameter difference:", full_model.count_params() - without_model.count_params())
print("Output shape:", full_model.output_shape)
