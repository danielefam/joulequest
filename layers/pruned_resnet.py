"""Canonical pruned ResNet-18 specifications derived from JouleNAS logs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


BLOCK_NAMES = tuple(
    f"layer{stage}.{block}"
    for stage in range(1, 5)
    for block in range(2)
)
BLOCK_STRIDES = (1, 1, 2, 1, 2, 1, 2, 1)


def _positive_channel(data, layer_name, field):
    try:
        value = data[layer_name][field]
    except KeyError as error:
        raise ValueError(f"Missing pruning field {layer_name}.{field}") from error
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{layer_name}.{field} must be a positive integer")
    return value


@dataclass(frozen=True)
class ShortcutSpec:
    in_channels: int
    out_channels: int


@dataclass(frozen=True)
class BlockSpec:
    name: str
    in_channels: int
    mid_channels: int
    out_channels: int
    stride: int
    shortcut: ShortcutSpec | None = None


@dataclass(frozen=True)
class OperationSpec:
    model_name: str
    count: int


@dataclass(frozen=True)
class PrunedResNet18Spec:
    stem_out: int
    blocks: tuple[BlockSpec, ...]
    classifier_in: int
    classifier_out: int

    @classmethod
    def from_pruning_log(cls, path):
        with Path(path).open(encoding="utf-8") as pruning_file:
            return cls.from_pruning_data(json.load(pruning_file))

    @classmethod
    def from_pruning_data(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Pruning log must contain a JSON object")

        stem_out = _positive_channel(data, "stem.conv", "active_output")
        previous_channels = stem_out
        blocks = []
        for name, stride in zip(BLOCK_NAMES, BLOCK_STRIDES):
            conv1 = f"{name}.conv1"
            conv2 = f"{name}.conv2"
            in_channels = _positive_channel(data, conv1, "active_input")
            mid_channels = _positive_channel(data, conv1, "active_output")
            conv2_input = _positive_channel(data, conv2, "active_input")
            out_channels = _positive_channel(data, conv2, "active_output")
            if in_channels != previous_channels:
                raise ValueError(
                    f"{conv1}.active_input={in_channels} does not match "
                    f"the previous output ({previous_channels})"
                )
            if conv2_input != mid_channels:
                raise ValueError(
                    f"{conv2}.active_input={conv2_input} does not match "
                    f"{conv1}.active_output ({mid_channels})"
                )

            shortcut_name = f"{name}.downsample.0"
            shortcut = None
            if shortcut_name in data:
                shortcut = ShortcutSpec(
                    _positive_channel(data, shortcut_name, "active_input"),
                    _positive_channel(data, shortcut_name, "active_output"),
                )
                if shortcut.in_channels != in_channels:
                    raise ValueError(
                        f"{shortcut_name}.active_input must match {conv1}.active_input"
                    )
                if shortcut.out_channels != out_channels:
                    raise ValueError(
                        f"{shortcut_name}.active_output must match {conv2}.active_output"
                    )

            blocks.append(
                BlockSpec(
                    name=name,
                    in_channels=in_channels,
                    mid_channels=mid_channels,
                    out_channels=out_channels,
                    stride=stride,
                    shortcut=shortcut,
                )
            )
            previous_channels = out_channels

        classifier_in = _positive_channel(data, "classifier", "active_input")
        classifier_out = _positive_channel(data, "classifier", "active_output")
        if classifier_in != previous_channels:
            raise ValueError(
                f"classifier.active_input={classifier_in} does not match "
                f"the final block output ({previous_channels})"
            )
        return cls(stem_out, tuple(blocks), classifier_in, classifier_out)

    @classmethod
    def from_config_dict(cls, config, classifier_out=10):
        blocks = []
        for index, (block, stride) in enumerate(
            zip(config["blocks"], BLOCK_STRIDES)
        ):
            shortcut_data = block.get("shortcut")
            shortcut = (
                ShortcutSpec(**shortcut_data) if shortcut_data is not None else None
            )
            blocks.append(
                BlockSpec(
                    name=BLOCK_NAMES[index],
                    in_channels=block["in_channels"],
                    mid_channels=block["mid_channels"],
                    out_channels=block["out_channels"],
                    stride=block.get("stride", stride),
                    shortcut=shortcut,
                )
            )
        return cls(
            stem_out=config["stem_out"],
            blocks=tuple(blocks),
            classifier_in=config["classifier_in"],
            classifier_out=classifier_out,
        )

    def to_dict(self):
        return {
            "schema_version": 1,
            "stem_out": self.stem_out,
            "blocks": [
                {
                    "name": block.name,
                    "in_channels": block.in_channels,
                    "mid_channels": block.mid_channels,
                    "out_channels": block.out_channels,
                    "stride": block.stride,
                    "shortcut": (
                        None
                        if block.shortcut is None
                        else {
                            "in_channels": block.shortcut.in_channels,
                            "out_channels": block.shortcut.out_channels,
                        }
                    ),
                }
                for block in self.blocks
            ],
            "classifier_in": self.classifier_in,
            "classifier_out": self.classifier_out,
        }

    def canonical_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def architecture_id(self):
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:16]

    def to_model_config(self):
        return {
            "stem_out": self.stem_out,
            "blocks": [
                {
                    "in_channels": block.in_channels,
                    "mid_channels": block.mid_channels,
                    "out_channels": block.out_channels,
                    "stride": block.stride,
                    "shortcut": (
                        None
                        if block.shortcut is None
                        else {
                            "in_channels": block.shortcut.in_channels,
                            "out_channels": block.shortcut.out_channels,
                        }
                    ),
                }
                for block in self.blocks
            ],
            "classifier_in": self.classifier_in,
            "shortcut_policy": "explicit",
        }

    def estimator_operations(self, image_size=32):
        """Return deduplicated Conv2d/Linear model names and call counts."""
        counts = Counter()
        counts[f"ResNetConv_3_{self.stem_out}_{image_size}_3_1_1.pt"] += 1
        spatial_size = image_size // 2  # CIFAR stem includes a 3x3 stride-2 pool.
        for block in self.blocks:
            counts[
                f"ResNetConv_{block.in_channels}_{block.mid_channels}_"
                f"{spatial_size}_3_{block.stride}_1.pt"
            ] += 1
            output_size = (spatial_size + 2 - 3) // block.stride + 1
            counts[
                f"ResNetConv_{block.mid_channels}_{block.out_channels}_"
                f"{output_size}_3_1_1.pt"
            ] += 1
            if block.shortcut is not None:
                counts[
                    f"ResNetConv_{block.shortcut.in_channels}_"
                    f"{block.shortcut.out_channels}_{spatial_size}_1_"
                    f"{block.stride}_0.pt"
                ] += 1
            spatial_size = output_size
        counts[f"Linear_{self.classifier_in}_{self.classifier_out}.pt"] += 1
        return tuple(
            OperationSpec(model_name, count)
            for model_name, count in sorted(counts.items())
        )