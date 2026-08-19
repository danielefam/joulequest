"""Query a generated energy lookup table from the command line."""

import argparse

from .lookup import EnergyLookup


def build_parser():
    parser = argparse.ArgumentParser(description="Estimate per-inference layer energy.")
    parser.add_argument("lookup", help="Generated energy_lookup_table.csv")
    parser.add_argument(
        "--out-of-range",
        choices=("error", "clamp", "extrapolate"),
        default="error",
    )
    subparsers = parser.add_subparsers(dest="layer_type", required=True)

    linear = subparsers.add_parser("linear")
    linear.add_argument("input_features", type=float)
    linear.add_argument("output_features", type=float)

    conv = subparsers.add_parser("conv")
    conv.add_argument("input_channels", type=float)
    conv.add_argument("output_channels", type=float)
    conv.add_argument("input_height", type=int)
    conv.add_argument("--input-width", type=int)
    conv.add_argument("--kernel-size", type=int, required=True)
    conv.add_argument("--stride", type=int, default=1)
    conv.add_argument("--padding", type=int, default=0)

    attention = subparsers.add_parser("attention")
    attention.add_argument("sequence_length", type=float)
    attention.add_argument("embed_dim", type=float)
    attention.add_argument("head_dim", type=float)
    attention.add_argument("--rotary", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    lookup = EnergyLookup(args.lookup, out_of_range=args.out_of_range)
    if args.layer_type == "linear":
        energy = lookup.linear(args.input_features, args.output_features)
    elif args.layer_type == "conv":
        energy = lookup.conv2d(
            args.input_channels,
            args.output_channels,
            args.input_height,
            args.input_width,
            kernel_size=args.kernel_size,
            stride=args.stride,
            padding=args.padding,
        )
    else:
        energy = lookup.attention(
            args.sequence_length,
            args.embed_dim,
            args.head_dim,
            rotary=args.rotary,
        )
    print(f"{float(energy.detach()):.12g} mJ/inference")


if __name__ == "__main__":
    main()