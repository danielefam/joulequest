#!/usr/bin/env python3
"""Emit a shell-readable measurement plan for one pruned ResNet-18 log."""

from __future__ import annotations

import argparse
from pathlib import Path

from layers.pruned_resnet import PrunedResNet18Spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pruning-log", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=32)
    args = parser.parse_args(argv)

    spec = PrunedResNet18Spec.from_pruning_log(args.pruning_log)
    print(f"META\t{spec.architecture_id}\t{spec.canonical_json()}")
    for operation in spec.estimator_operations(args.image_size):
        print(f"MODEL\t{operation.model_name}\t{operation.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())