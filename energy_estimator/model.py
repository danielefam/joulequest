"""Model-level energy accumulation for supported PyTorch layers."""

from collections.abc import Mapping

import torch
from torch import nn

from .lookup import EnergyLookup


def _effective_dimension(mask_spec, key, default):
    if isinstance(mask_spec, torch.Tensor):
        return mask_spec.sum() if key == "output" else default
    if isinstance(mask_spec, Mapping) and key in mask_spec:
        value = mask_spec[key]
        return value.sum() if isinstance(value, torch.Tensor) and value.ndim else value
    return default


def _scalar_parameter(value, name):
    if isinstance(value, tuple):
        if len(value) != 2 or value[0] != value[1]:
            raise ValueError(f"Only square Conv2d {name} values are supported: {value}")
        return value[0]
    return value


def _spatial_shape(value):
    if len(value) < 2:
        raise ValueError("A Conv2d input shape must contain height and width")
    return value[-2], value[-1]


def estimate_model_energy(
    model,
    lookup,
    *,
    masks=None,
    input_shapes=None,
    module_names=None,
    skip_unsupported=False,
):
    """Sum estimated Linear and Conv2d energy recursively.

    ``masks`` maps qualified module names either to an output mask tensor or to
    ``{"input": input_mask, "output": output_mask}``. Explicit input masks are
    required when graph connectivity changes dimensions; this function does not
    infer data flow from module registration order.
    """

    if not isinstance(lookup, EnergyLookup):
        lookup = EnergyLookup(lookup)
    masks = {} if masks is None else masks
    input_shapes = {} if input_shapes is None else input_shapes
    selected = None if module_names is None else set(module_names)
    layers = {}
    total = None

    for name, module in model.named_modules():
        if not name or (selected is not None and name not in selected):
            continue
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue

        mask_spec = masks.get(name)
        try:
            if isinstance(module, nn.Linear):
                effective_input = _effective_dimension(
                    mask_spec, "input", module.in_features
                )
                effective_output = _effective_dimension(
                    mask_spec, "output", module.out_features
                )
                energy = lookup.linear(effective_input, effective_output)
            else:
                if module.groups != 1 or module.dilation != (1, 1):
                    raise ValueError(
                        "Grouped or dilated Conv2d measurements are not available"
                    )
                if name not in input_shapes:
                    raise ValueError(f"Missing input shape for Conv2d module '{name}'")
                input_height, input_width = _spatial_shape(input_shapes[name])
                effective_input = _effective_dimension(
                    mask_spec, "input", module.in_channels
                )
                effective_output = _effective_dimension(
                    mask_spec, "output", module.out_channels
                )
                energy = lookup.conv2d(
                    effective_input,
                    effective_output,
                    input_height,
                    input_width,
                    kernel_size=_scalar_parameter(module.kernel_size, "kernel_size"),
                    stride=_scalar_parameter(module.stride, "stride"),
                    padding=_scalar_parameter(module.padding, "padding"),
                )
        except ValueError:
            if skip_unsupported:
                continue
            raise

        total = energy if total is None else total + energy
        layers[name] = {
            "module_type": type(module).__name__,
            "effective_input": effective_input,
            "effective_output": effective_output,
            "energy_mJ": energy,
        }

    if total is None:
        total = torch.tensor(0.0, dtype=lookup.dtype)
    return {"layers": layers, "total_energy_mJ": total}