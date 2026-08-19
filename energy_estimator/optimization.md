# Energy-estimator performance optimization

This document records the performance problems found in the original estimator,
the implemented solutions, measured results, and the remaining constraints. The
mathematical interpolation and energy units have not changed.

## Intended training path

Create the lookup and model regularizer once, after placing the model on its
training device:

```python
from energy_estimator import EnergyLookup, ModelEnergyRegularizer

lookup = EnergyLookup(
    "measurements/Plot/pi5/energy_lookup_table.csv",
    out_of_range="error",
)
energy_regularizer = ModelEnergyRegularizer(
    model,
    lookup,
    input_shapes={
        "features.0": (batch_size, input_channels, input_height, input_width),
    },
)
```

Reuse the prepared regularizer in every training step:

```python
energy_mj = energy_regularizer(
    {
        "features.0": {
            "input": input_channel_probabilities,
            "output": output_channel_probabilities,
        },
        "classifier": output_feature_probabilities,
    }
)
loss = task_loss + lambda_energy * energy_mj
```

The returned value is one scalar PyTorch tensor in mJ per inference. It remains
connected to the soft masks, so the final `loss.backward()` propagates through
the sum of all layer estimates.

Use `energy_regularizer.estimate(masks)` only for occasional logging. It also
constructs the per-layer result dictionary and is therefore not the minimal hot
path.

## Problems and solutions

### Lookup grids were rebuilt for every layer

**Problem:** Every query filtered a Pandas DataFrame, computed unique axes,
allocated a dense tensor, built Python index dictionaries, and copied every
measurement into the tensor. This repeated static work during every training
step and for every layer.

**Solution:** `EnergyLookup` now builds and validates every Linear, Conv2d, and
attention grid once in its constructor. Queries select an immutable prepared
grid by a small key:

```text
("linear",)
("conv", kernel_size, stride, padding)
("attention",)
("rotaryattention",)
```

Pandas is no longer used in the query path.

### Lookup tensors were repeatedly transferred

**Problem:** Axes and energy tensors were moved to the coordinate device on
every query.

**Solution:** Prepared grids are cached per device. The first query on a device
creates its tensor copy; subsequent queries reuse it. This keeps the static
lookup values outside the repeated training work.

### Interpolation corners were evaluated with Python loops

**Problem:** Bilinear and trilinear interpolation visited every corner using
nested Python logic and allocated many scalar tensors.

**Solution:** Corner bit patterns are cached by interpolation rank and device.
All corner indices, weights, values, and the final weighted sum are evaluated as
vectorized PyTorch operations. PyTorch executes these operations in its C++ or
CUDA backend while autograd records the interpolation weights.

### Every layer issued a separate interpolation query

**Problem:** Summing energy over many layers repeated Python dispatch and small
tensor operations. Small independent operations are particularly inefficient on
accelerators.

**Solution:** The lookup exposes `linear_batch`, `conv2d_batch`, and
`attention_batch`. `ModelEnergyRegularizer` groups layers that share a grid and
evaluates each group in one batched interpolation. All returned vectors are
concatenated and reduced with one tensor sum.

### The model was traversed every training step

**Problem:** The diagnostic `estimate_model_energy` function recursively scanned
`model.named_modules()` whenever it was called.

**Solution:** `ModelEnergyRegularizer` scans once during construction and stores
only the required layer names, dimensions, convolution geometry, and input
shapes. Its `forward` method consumes current masks without traversing the model.
The existing `estimate_model_energy` function remains available for compatibility
and one-off diagnostics, but it constructs a prepared regularizer on each call.

### Bounds checks could synchronize GPU and CPU

**Problem:** Python conversion of tensor comparisons can force device
synchronization.

**Solution:** `clamp` and `extrapolate` avoid host-side bounds checks on normal
multi-point axes. Strict `error` mode still synchronizes when it must decide
whether to raise a Python exception. Complete grids skip runtime NaN-corner
checks; sparse grids retain them to prevent unsupported estimates.

## Performance measurements

Measurements were taken on the project CPU environment with the Pi 5 lookup
table. They measure estimator forward time, not model inference or backward.
Absolute times depend on hardware and PyTorch version.

| Workload | Before | After | Speedup |
|---|---:|---:|---:|
| One Linear query | 1.766 ms | 0.146 ms | 12.1x |
| One Conv2d query | 2.612 ms | 0.176 ms | 14.8x |
| Sum of 64 differentiable Linear layers | 16.892 ms | 0.885 ms | 19.1x |

For the 64-layer test, scalar and batched totals were exactly equal under
`torch.allclose`, and every mask received a finite gradient.

## Why this is not rewritten in C

The expensive work was repeated data preparation and Python dispatch, not the
multilinear formula itself. After preparation and vectorization, interpolation
runs through PyTorch's existing optimized C++/CUDA kernels. A custom C or C++
extension would add build, portability, and custom-autograd maintenance costs
without addressing the original data-path problem.

A native extension should only be reconsidered after profiling the optimized
training workload on the target accelerator and proving that this estimator is
still a meaningful bottleneck.

## Correctness and remaining constraints

- Energy remains measured in mJ per inference.
- Exact measured coordinates return the measured values.
- Soft effective dimensions remain differentiable.
- Missing interpolation corners still raise an error.
- Kernel size, stride, and padding remain discrete Conv2d keys.
- Grouped and dilated convolutions remain unsupported without measurements.
- Strict `error` bounds and sparse-grid validation can still synchronize an
  accelerator because raising a Python exception requires a host decision.
- `ModelEnergyRegularizer` automatically aggregates `nn.Linear` and `nn.Conv2d`.
  Custom prunable attention or MLP components should call the batch lookup APIs
  from their own existing regularizer integration to avoid double counting their
  internal Linear projections.
- Diagnostic dictionaries keep tensor values so logging code must explicitly
  detach values outside the loss path.

The focused tests compare scalar and batched results, verify gradients, exercise
missing cells and bounds policies, and compare the prepared regularizer with the
legacy diagnostic API. The complete repository test suite is the final
regression gate.