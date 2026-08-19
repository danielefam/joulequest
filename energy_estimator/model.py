"""Model-level energy accumulation for supported PyTorch layers."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from .lookup import EnergyLookup


@dataclass(frozen=True)
class _LinearSpec:
    name: str
    input_features: int
    output_features: int


@dataclass(frozen=True)
class _ConvSpec:
    name: str
    input_channels: int
    output_channels: int
    input_height: int
    input_width: int
    kernel_size: int
    stride: int
    padding: int


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


def _first_tensor_device(value):
    if isinstance(value, torch.Tensor):
        return value.device
    if isinstance(value, Mapping):
        for child in value.values():
            device = _first_tensor_device(child)
            if device is not None:
                return device
    return None


def _stack_scalars(values, *, device, dtype):
    tensors = []
    for value in values:
        if isinstance(value, torch.Tensor):
            tensors.append(value.to(device=device, dtype=dtype).reshape(()))
        else:
            tensors.append(torch.tensor(float(value), device=device, dtype=dtype))
    return torch.stack(tensors)


class ModelEnergyRegularizer(nn.Module):
    """Prepared, batched energy regularizer for repeated training steps."""

    def __init__(
        self,
        model,
        lookup,
        *,
        input_shapes=None,
        module_names=None,
        skip_unsupported=False,
    ):
        super().__init__()
        self.lookup = lookup if isinstance(lookup, EnergyLookup) else EnergyLookup(lookup)
        self.skip_unsupported = skip_unsupported
        input_shapes = {} if input_shapes is None else input_shapes
        selected = None if module_names is None else set(module_names)
        self._linear_specs = []
        self._conv_specs = {}

        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
        self.register_buffer(
            "_device_anchor",
            torch.empty(0, device=model_device),
            persistent=False,
        )

        for name, module in model.named_modules():
            if not name or (selected is not None and name not in selected):
                continue
            if isinstance(module, nn.Linear):
                self._linear_specs.append(
                    _LinearSpec(name, module.in_features, module.out_features)
                )
                continue
            if not isinstance(module, nn.Conv2d):
                continue

            try:
                if module.groups != 1 or module.dilation != (1, 1):
                    raise ValueError(
                        "Grouped or dilated Conv2d measurements are not available"
                    )
                if name not in input_shapes:
                    raise ValueError(f"Missing input shape for Conv2d module '{name}'")
                input_height, input_width = _spatial_shape(input_shapes[name])
                kernel_size = _scalar_parameter(module.kernel_size, "kernel_size")
                stride = _scalar_parameter(module.stride, "stride")
                padding = _scalar_parameter(module.padding, "padding")
            except ValueError:
                if skip_unsupported:
                    continue
                raise

            key = (kernel_size, stride, padding)
            self._conv_specs.setdefault(key, []).append(
                _ConvSpec(
                    name,
                    module.in_channels,
                    module.out_channels,
                    input_height,
                    input_width,
                    kernel_size,
                    stride,
                    padding,
                )
            )

    def _evaluate(self, masks, *, include_details):
        masks = {} if masks is None else masks
        device = _first_tensor_device(masks) or self._device_anchor.device
        energy_batches = []
        layers = {}

        if self._linear_specs:
            linear_specs = self._linear_specs
            effective_inputs = [
                _effective_dimension(
                    masks.get(spec.name), "input", spec.input_features
                )
                for spec in self._linear_specs
            ]
            effective_outputs = [
                _effective_dimension(
                    masks.get(spec.name), "output", spec.output_features
                )
                for spec in self._linear_specs
            ]
            try:
                energies = self.lookup.linear_batch(
                    _stack_scalars(effective_inputs, device=device, dtype=self.lookup.dtype),
                    _stack_scalars(effective_outputs, device=device, dtype=self.lookup.dtype),
                )
            except ValueError:
                if not self.skip_unsupported:
                    raise
                supported = []
                scalar_energies = []
                for spec, effective_input, effective_output in zip(
                    linear_specs, effective_inputs, effective_outputs
                ):
                    try:
                        energy = self.lookup.linear(
                            effective_input, effective_output
                        )
                    except ValueError:
                        continue
                    supported.append((spec, effective_input, effective_output))
                    scalar_energies.append(energy)
                linear_specs = [item[0] for item in supported]
                effective_inputs = [item[1] for item in supported]
                effective_outputs = [item[2] for item in supported]
                energies = (
                    torch.stack(scalar_energies) if scalar_energies else None
                )
            if energies is not None:
                energy_batches.append(energies)
                if include_details:
                    for index, spec in enumerate(linear_specs):
                        layers[spec.name] = {
                            "module_type": "Linear",
                            "effective_input": effective_inputs[index],
                            "effective_output": effective_outputs[index],
                            "energy_mJ": energies[index],
                        }

        for (kernel_size, stride, padding), specs in self._conv_specs.items():
            conv_specs = specs
            effective_inputs = [
                _effective_dimension(
                    masks.get(spec.name), "input", spec.input_channels
                )
                for spec in specs
            ]
            effective_outputs = [
                _effective_dimension(
                    masks.get(spec.name), "output", spec.output_channels
                )
                for spec in specs
            ]
            try:
                energies = self.lookup.conv2d_batch(
                    _stack_scalars(effective_inputs, device=device, dtype=self.lookup.dtype),
                    _stack_scalars(effective_outputs, device=device, dtype=self.lookup.dtype),
                    torch.tensor(
                        [spec.input_height for spec in specs], device=device
                    ),
                    torch.tensor([spec.input_width for spec in specs], device=device),
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            except ValueError:
                if not self.skip_unsupported:
                    raise
                supported = []
                scalar_energies = []
                for spec, effective_input, effective_output in zip(
                    conv_specs, effective_inputs, effective_outputs
                ):
                    try:
                        energy = self.lookup.conv2d(
                            effective_input,
                            effective_output,
                            spec.input_height,
                            spec.input_width,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                        )
                    except ValueError:
                        continue
                    supported.append((spec, effective_input, effective_output))
                    scalar_energies.append(energy)
                conv_specs = [item[0] for item in supported]
                effective_inputs = [item[1] for item in supported]
                effective_outputs = [item[2] for item in supported]
                energies = (
                    torch.stack(scalar_energies) if scalar_energies else None
                )
            if energies is not None:
                energy_batches.append(energies)
                if include_details:
                    for index, spec in enumerate(conv_specs):
                        layers[spec.name] = {
                            "module_type": "Conv2d",
                            "effective_input": effective_inputs[index],
                            "effective_output": effective_outputs[index],
                            "energy_mJ": energies[index],
                        }

        if energy_batches:
            total = torch.cat(energy_batches).sum()
        else:
            total = torch.zeros((), device=device, dtype=self.lookup.dtype)
        if include_details:
            return {"layers": layers, "total_energy_mJ": total}
        return total

    def forward(self, masks=None):
        """Return the scalar differentiable energy term for a training loss."""

        return self._evaluate(masks, include_details=False)

    def estimate(self, masks=None):
        """Return total and per-layer estimates for logging or diagnostics."""

        return self._evaluate(masks, include_details=True)


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

    regularizer = ModelEnergyRegularizer(
        model,
        lookup,
        input_shapes=input_shapes,
        module_names=module_names,
        skip_unsupported=skip_unsupported,
    )
    return regularizer.estimate(masks)