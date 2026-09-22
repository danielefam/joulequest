#!/usr/bin/env python3
"""Materialize and inspect a pruned ResNet-18 model from a JouleNAS pruning log."""

from __future__ import annotations

import argparse
import pprint
import sys
from pathlib import Path

from layers.pruned_resnet import PrunedResNet18Spec


def parse_pruning_log(path: Path) -> dict:
    return PrunedResNet18Spec.from_pruning_log(path).to_model_config()


def format_config_python(cfg: dict) -> str:
    return "DEFAULT_PRUNED_CONFIG_ORIN_BS32 = " + pprint.pformat(
        cfg,
        sort_dicts=False,
        width=100,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize a pruned ResNet-18 model from JouleNAS pruning log.")
    parser.add_argument("--pruning-log", required=True, type=Path, help="Path to JouleNAS pruning log JSON")
    parser.add_argument("--update-resnet", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--test-forward", action="store_true", help="Run a test forward pass with batch size 32")
    args = parser.parse_args()

    if not args.pruning_log.is_file():
        print(f"Error: File not found: {args.pruning_log}", file=sys.stderr)
        return 1

    spec = PrunedResNet18Spec.from_pruning_log(args.pruning_log)
    cfg = spec.to_model_config()
    py_code = format_config_python(cfg)
    print("======================================================================")
    print(" Extracted Pruned Architecture Configuration:")
    print("======================================================================")
    print(py_code)
    print(f"Architecture ID: {spec.architecture_id}")
    print("======================================================================")

    if args.update_resnet:
        print(
            "Error: --update-resnet was removed. Use run_measurement_campaign.sh "
            "--suite pruned_resnet18 --pruning-log PATH.",
            file=sys.stderr,
        )
        return 2

    if args.test_forward:
        import torch
        from layers.resnet import PrunedCifarResNet18
        model = PrunedCifarResNet18(num_classes=10, cifar_stem=True, config=cfg)
        x = torch.randn(32, 3, 32, 32)
        out = model(x)
        print(f"Forward test successful! Input: {x.shape} -> Output: {out.shape}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

