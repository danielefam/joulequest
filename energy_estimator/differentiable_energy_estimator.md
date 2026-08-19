# Differentiable energy estimator

The `energy_estimator` package turns the CSV produced by
`energy_estimator/build_energy_lookup_table.py` into differentiable PyTorch
energy estimates. Measurement acquisition remains independent of training: the
package only reads completed, processed measurements.

All estimates use **mJ per inference**, matching `energy_mean_mJ` in the CSV.
Tables from different boards or software configurations must not be combined.

## Generate a lookup table

```bash
python -m energy_estimator.build_energy_lookup_table \
  measurements/Plot/BOARD/summary.csv \
  --output measurements/Plot/BOARD/energy_lookup_table.csv
```

The builder recognizes these measured model names:

```text
Linear_<input_features>_<output_features>.pt
Conv_<input_channels>_<image_size>_<kernel>_<padding>_<output_channels>.pt
ResNetConv_<input_channels>_<output_channels>_<image_size>_<kernel>_<stride>_<padding>.pt
Attention_<sequence_length>_<embed_dim>_<num_heads>.pt
RotaryAttention_<sequence_length>_<embed_dim>_<num_heads>.pt
```

For attention, `head_dim` is derived as `embed_dim / num_heads`. Invalid names
whose dimensions are not divisible are ignored.

## Interpolation

- Linear uses bilinear interpolation over input and output features.
- Conv2d holds kernel size, stride, and padding fixed, then uses trilinear
  interpolation over input channels, output channels, and output spatial area.
- Attention uses trilinear interpolation over sequence length, embedding
  dimension, and head dimension.

Output spatial area is derived from the measured input size and convolution
geometry. This follows executed convolution work more closely than interpolating
directly over image side length. Rectangular queries are supported even though
the current campaign uses square measured images.

Every corner required by an interpolation cell must have a measurement. Missing
corners raise an error instead of silently fabricating values. An axis containing
one measured coordinate is usable only at that coordinate unless clamping is
enabled. The default out-of-range policy is `error`; `clamp` and `extrapolate`
must be selected explicitly.

## Layer queries

```python
import torch

from energy_estimator import EnergyLookup

lookup = EnergyLookup("measurements/Plot/pi5/energy_lookup_table.csv")

mask_logits = torch.zeros(128, requires_grad=True)
effective_outputs = torch.sigmoid(mask_logits).sum()
energy_mj = lookup.linear(64, effective_outputs)
energy_mj.backward()
```

The coordinates may be scalar tensors. Interpolation-cell selection is discrete,
but interpolation weights remain connected to autograd. Do not round, detach, or
convert effective dimensions to Python values in the regularization path.

Conv2d and attention queries use:

```python
conv_energy = lookup.conv2d(
    effective_input_channels,
    effective_output_channels,
    input_height=224,
    input_width=224,
    kernel_size=3,
    stride=1,
    padding=1,
)

attention_energy = lookup.attention(
    sequence_length=128,
    embed_dim=effective_embed_dim,
    head_dim=effective_head_dim,
)
```

## Model aggregation

For training, construct `ModelEnergyRegularizer` once. It recursively finds
`nn.Linear` and `nn.Conv2d` modules, caches their metadata, and batches compatible
layers. Masks can be an output mask tensor or an explicit input/output mapping:

```python
from energy_estimator import ModelEnergyRegularizer

regularizer = ModelEnergyRegularizer(
    model,
    lookup,
  input_shapes={"features.0": (1, 64, 224, 224)},
)
energy_regularizer = regularizer(
    masks={
        "features.0": {
            "input": input_channel_probabilities,
            "output": output_channel_probabilities,
        },
        "classifier": output_feature_probabilities,
    },
)
loss = task_loss + lambda_energy * energy_regularizer
```

    Call `regularizer.estimate(masks)` when per-layer values are needed for logging.
    The compatibility function `estimate_model_energy` constructs and scans a new
    regularizer on every call, so it is intended for diagnostics rather than the
    training hot path.

Input masks are explicit because module registration order does not describe
model data flow, especially around residual branches. The estimator does not
guess mask propagation. Conv input shapes are also explicit because an
`nn.Conv2d` module does not store the runtime image size.

Grouped and dilated convolutions are rejected until matching experiments exist.
Use `module_names` to avoid counting Linear projections already represented by a
measured aggregate attention or MLP component.

Batch APIs are also available for custom pruning integrations:
`linear_batch`, `conv2d_batch`, and `attention_batch`.

## Command line

```bash
python -m energy_estimator measurements/Plot/pi5/energy_lookup_table.csv \
  linear 64 64

python -m energy_estimator measurements/Plot/pi5/energy_lookup_table.csv \
  conv 1 1 64 --kernel-size 3 --padding 1
```

Attention queries become available after regenerating a table containing the
new attention campaign results.