#!/usr/bin/env python3
"""Materialize and inspect a pruned ResNet-18 model from a JouleNAS pruning log."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_pruning_log(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stem_out = data["stem.conv"]["active_output"]
    blocks = [
        {
            "in_channels": data["layer1.0.conv1"]["active_input"],
            "mid_channels": data["layer1.0.conv1"]["active_output"],
            "out_channels": data["layer1.0.conv2"]["active_output"],
            "stride": 1,
        },
        {
            "in_channels": data["layer1.1.conv1"]["active_input"],
            "mid_channels": data["layer1.1.conv1"]["active_output"],
            "out_channels": data["layer1.1.conv2"]["active_output"],
            "stride": 1,
        },
        {
            "in_channels": data["layer2.0.conv1"]["active_input"],
            "mid_channels": data["layer2.0.conv1"]["active_output"],
            "out_channels": data["layer2.0.conv2"]["active_output"],
            "stride": 2,
        },
        {
            "in_channels": data["layer2.1.conv1"]["active_input"],
            "mid_channels": data["layer2.1.conv1"]["active_output"],
            "out_channels": data["layer2.1.conv2"]["active_output"],
            "stride": 1,
        },
        {
            "in_channels": data["layer3.0.conv1"]["active_input"],
            "mid_channels": data["layer3.0.conv1"]["active_output"],
            "out_channels": data["layer3.0.conv2"]["active_output"],
            "stride": 2,
        },
        {
            "in_channels": data["layer3.1.conv1"]["active_input"],
            "mid_channels": data["layer3.1.conv1"]["active_output"],
            "out_channels": data["layer3.1.conv2"]["active_output"],
            "stride": 1,
        },
        {
            "in_channels": data["layer4.0.conv1"]["active_input"],
            "mid_channels": data["layer4.0.conv1"]["active_output"],
            "out_channels": data["layer4.0.conv2"]["active_output"],
            "stride": 2,
        },
        {
            "in_channels": data["layer4.1.conv1"]["active_input"],
            "mid_channels": data["layer4.1.conv1"]["active_output"],
            "out_channels": data["layer4.1.conv2"]["active_output"],
            "stride": 1,
        },
    ]
    classifier_in = data["classifier"]["active_input"]
    return {
        "stem_out": stem_out,
        "blocks": blocks,
        "classifier_in": classifier_in,
    }


def format_config_python(cfg: dict) -> str:
    lines = ['DEFAULT_PRUNED_CONFIG_ORIN_BS32 = {', f'    "stem_out": {cfg["stem_out"]},', '    "blocks": [']
    for idx, b in enumerate(cfg["blocks"]):
        stage = (idx // 2) + 1
        blk = idx % 2
        lines.append(f'        # stage {stage} (layer{stage}.{blk})')
        lines.append(f'        {{"in_channels": {b["in_channels"]}, "mid_channels": {b["mid_channels"]}, "out_channels": {b["out_channels"]}, "stride": {b["stride"]}}},')
    lines.append('    ],')
    lines.append(f'    "classifier_in": {cfg["classifier_in"]},')
    lines.append('}')
    return '\n'.join(lines)


def update_layers_resnet(resnet_path: Path, new_code: str) -> None:
    content = resnet_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r'DEFAULT_PRUNED_CONFIG_ORIN_BS32\s*=\s*(?:DEFAULT_PRUNED_CONFIG_ORIN\.copy\(\)|\{[^}]+\})',
        re.DOTALL,
    )
    if pattern.search(content):
        updated = pattern.sub(new_code, content)
        resnet_path.write_text(updated, encoding="utf-8")
        print(f"Updated {resnet_path} with new DEFAULT_PRUNED_CONFIG_ORIN_BS32")
    else:
        print(f"Warning: could not locate DEFAULT_PRUNED_CONFIG_ORIN_BS32 in {resnet_path}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize a pruned ResNet-18 model from JouleNAS pruning log.")
    parser.add_argument("--pruning-log", required=True, type=Path, help="Path to JouleNAS pruning log JSON")
    parser.add_argument("--update-resnet", action="store_true", help="Automatically update layers/resnet.py")
    parser.add_argument("--test-forward", action="store_true", help="Run a test forward pass with batch size 32")
    args = parser.parse_args()

    if not args.pruning_log.is_file():
        print(f"Error: File not found: {args.pruning_log}", file=sys.stderr)
        return 1

    cfg = parse_pruning_log(args.pruning_log)
    py_code = format_config_python(cfg)
    print("======================================================================")
    print(" Extracted Pruned Architecture Configuration:")
    print("======================================================================")
    print(py_code)
    print("======================================================================")

    if args.update_resnet:
        this_dir = Path(__file__).resolve().parent
        resnet_path = this_dir / "layers" / "resnet.py"
        update_layers_resnet(resnet_path, py_code)

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

