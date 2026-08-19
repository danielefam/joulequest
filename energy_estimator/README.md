# Energy estimator

This directory contains the energy-estimation framework used by energyBANERA.
It converts processed hardware measurements into differentiable estimates of
energy per inference for Linear, Conv2d, and attention layers.

The initial idea was inspired by [energy_estimator](https://github.com/aissa0803/energy_estimator); the code in this directory has been substantially rewritten for energyBANERA.

## Improvements over the upstream version

- Uses energyBANERA's processed CSV summaries directly; Excel power matrices are
  not required.
- Builds lookup tables from repeated measurements, retaining the measurement
  count, mean energy, and standard deviation for every exact configuration.
- Extends convolution estimation from two-dimensional channel interpolation to
  trilinear interpolation over input channels, output channels, and derived
  output spatial area. This accounts for changing image resolution while
  keeping kernel size, stride, and padding as discrete configuration keys.
- Adds measured Attention and RotaryAttention configurations, interpolated over
  sequence length, embedding dimension, and head dimension.
- Recursively aggregates supported modules and requires explicit masks and
  Conv2d input shapes, avoiding incorrect dimension propagation through
  residual or non-sequential model graphs.
- Rejects incomplete interpolation cells, unsupported Conv2d geometry, and
  out-of-range queries by default instead of silently estimating unsupported
  configurations.
- Precomputes and caches lookup grids, vectorizes interpolation corners, and
  batches compatible layers through `ModelEnergyRegularizer` for repeated
  training-step evaluation.

## Workflow

1. Run and process measurements for one compatible board.
2. Build a lookup table from the processed summaries:

   ```bash
   python -m energy_estimator.build_energy_lookup_table \
     measurements/Plot/pi5/summary.csv \
     --output measurements/Plot/pi5/energy_lookup_table.csv
   ```

3. Query the table or load it from Python:

   ```bash
   python -m energy_estimator \
     measurements/Plot/pi5/energy_lookup_table.csv \
     linear 64 64
   ```

   ```python
   from energy_estimator import EnergyLookup

   lookup = EnergyLookup("measurements/Plot/pi5/energy_lookup_table.csv")
   energy_mj = lookup.linear(64, 64)
   ```

## Supported measurements

The builder recognizes model filenames in these formats:

```text
Linear_<input_features>_<output_features>.pt
Conv_<input_channels>_<image_size>_<kernel>_<padding>_<output_channels>.pt
ResNetConv_<input_channels>_<output_channels>_<image_size>_<kernel>_<stride>_<padding>.pt
Attention_<sequence_length>_<embed_dim>_<num_heads>.pt
RotaryAttention_<sequence_length>_<embed_dim>_<num_heads>.pt
```

Only completed, quality-approved rows with numeric `energy_mean_mJ` values are
included. Measurements from different boards or incompatible configurations
must be written to separate lookup tables.

## Interpolation

- Linear uses bilinear interpolation over input and output features.
- Conv2d uses trilinear interpolation over input channels, output channels, and
  output spatial area. Kernel size, stride, and padding remain fixed.
- Attention uses trilinear interpolation over sequence length, embedding
  dimension, and head dimension.

The default policy rejects out-of-range coordinates and incomplete interpolation
cells. Use `EnergyLookup(path, out_of_range="clamp")` when clamping is explicitly
preferred. Coordinates may be differentiable PyTorch scalar tensors, so the
returned energy can be used as a training regularizer.

See [the detailed design document](differentiable_energy_estimator.md) for model
aggregation, masks, and NAS integration.

See [the optimization report](optimization.md) for the original performance
problems, implemented solutions, benchmarks, and recommended loss integration.