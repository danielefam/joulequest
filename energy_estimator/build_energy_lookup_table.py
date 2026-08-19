"""Build an energy lookup table from processed measurement summaries."""

import argparse
import re
from pathlib import Path

import pandas as pd


LINEAR_PATTERN = re.compile(r"^Linear_(\d+)_(\d+)$", re.IGNORECASE)
CONV_PATTERN = re.compile(
    r"^Conv_(\d+)_(\d+)_(\d+)_(\d+)(?:_(\d+))?$", re.IGNORECASE
)
RESNET_CONV_PATTERN = re.compile(
    r"^ResNetConv_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)$",
    re.IGNORECASE,
)
ATTENTION_PATTERN = re.compile(
    r"^(RotaryAttention|Attention)_(\d+)_(\d+)_(\d+)$",
    re.IGNORECASE,
)


def _dimensions(layer_type, **values):
    dimensions = {
        "layer_type": layer_type,
        "input_features": None,
        "output_features": None,
        "input_channels": None,
        "output_channels": None,
        "input_image_size": None,
        "kernel_size": None,
        "stride": None,
        "padding": None,
        "sequence_length": None,
        "embed_dim": None,
        "num_heads": None,
        "head_dim": None,
    }
    dimensions.update(values)
    return dimensions


def _parse_layer(model_path):
    """Return lookup dimensions encoded in a measured model filename."""

    stem = Path(str(model_path)).stem
    linear_match = LINEAR_PATTERN.fullmatch(stem)
    if linear_match:
        input_features, output_features = map(int, linear_match.groups())
        return _dimensions(
            "linear",
            input_features=input_features,
            output_features=output_features,
        )

    conv_match = CONV_PATTERN.fullmatch(stem)
    if conv_match:
        input_channels, image_size, kernel_size, padding, output_channels = (
            conv_match.groups()
        )
        return _dimensions(
            "conv",
            input_channels=int(input_channels),
            output_channels=int(output_channels or 1),
            input_image_size=int(image_size),
            kernel_size=int(kernel_size),
            stride=1,
            padding=int(padding),
        )

    resnet_conv_match = RESNET_CONV_PATTERN.fullmatch(stem)
    if resnet_conv_match:
        (
            input_channels,
            output_channels,
            image_size,
            kernel_size,
            stride,
            padding,
        ) = map(int, resnet_conv_match.groups())
        return _dimensions(
            "conv",
            input_channels=input_channels,
            output_channels=output_channels,
            input_image_size=image_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    attention_match = ATTENTION_PATTERN.fullmatch(stem)
    if attention_match:
        layer_name, sequence_length, embed_dim, num_heads = attention_match.groups()
        sequence_length, embed_dim, num_heads = map(
            int, (sequence_length, embed_dim, num_heads)
        )
        if embed_dim % num_heads != 0:
            return None
        layer_type = (
            "rotaryattention" if layer_name.lower().startswith("rotary") else "attention"
        )
        return _dimensions(
            layer_type,
            sequence_length=sequence_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=embed_dim // num_heads,
        )
    return None


def build_lookup_table(summary_paths):
    """Aggregate valid measured energy values by exact layer configuration."""

    measurements = []
    for summary_path in summary_paths:
        data = pd.read_csv(summary_path)
        required_columns = {"model_path", "energy_mean_mJ"}
        missing_columns = required_columns.difference(data.columns)
        if missing_columns:
            raise ValueError(
                f"{summary_path} is missing columns: "
                f"{', '.join(sorted(missing_columns))}"
            )
        if "status" in data:
            data = data[data["status"].eq("COMPLETE")]
        if "quality_status" in data:
            data = data[data["quality_status"].eq("OK")]

        for row in data.to_dict(orient="records"):
            energy_mj = pd.to_numeric(row["energy_mean_mJ"], errors="coerce")
            dimensions = _parse_layer(row["model_path"])
            if dimensions is not None and pd.notna(energy_mj):
                measurements.append({**dimensions, "energy_mean_mJ": energy_mj})

    columns = [
        "layer_type",
        "input_features",
        "output_features",
        "input_channels",
        "output_channels",
        "input_image_size",
        "kernel_size",
        "stride",
        "padding",
        "sequence_length",
        "embed_dim",
        "num_heads",
        "head_dim",
        "measurement_count",
        "energy_mean_mJ",
        "energy_stddev_mJ",
    ]
    if not measurements:
        return pd.DataFrame(columns=columns)

    dimensions = columns[:13]
    lookup = (
        pd.DataFrame(measurements)
        .groupby(dimensions, dropna=False, as_index=False)["energy_mean_mJ"]
        .agg(measurement_count="count", energy_mean_mJ="mean", energy_stddev_mJ="std")
        .sort_values(dimensions, kind="stable")
        .reset_index(drop=True)
    )
    return lookup[columns]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create a linear/convolution/attention energy lookup table."
    )
    parser.add_argument(
        "summary_paths",
        type=Path,
        nargs="+",
        help="Processed summary.csv files for one compatible board/configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("measurements/energy_lookup_table.csv"),
        help="CSV lookup-table destination.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    lookup = build_lookup_table(args.summary_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(args.output, index=False)
    print(f"Wrote {len(lookup)} lookup rows to {args.output}")


if __name__ == "__main__":
    main()