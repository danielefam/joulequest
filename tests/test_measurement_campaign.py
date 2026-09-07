import os
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_default_linear_conv_campaign_preserves_original_and_adds_pruning_points(
    tmp_path,
):
    environment = os.environ.copy()
    environment.update(
        {
            "CONNECTION_CONFIG": str(PROJECT_ROOT / "measurement_hosts.example.json"),
            "OUTPUT_ROOT": str(tmp_path),
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(PROJECT_ROOT / "run_measurement_campaign.sh"),
            "--board",
            "dryrun",
            "--suite",
            "linear,conv",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    model_paths = [
        match.group(1)
        for line in result.stdout.splitlines()
        if (match := re.match(r"^\[\d+/\d+\] (\S+)$", line))
    ]
    linear_pattern = re.compile(r"Linear_(\d+)_(\d+)\.pt$")
    conv_pattern = re.compile(r"Conv_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.pt$")
    linear_points = {
        tuple(map(int, match.groups()))
        for path in model_paths
        if (match := linear_pattern.search(path))
    }
    conv_points = {
        tuple(map(int, match.groups()))
        for path in model_paths
        if (match := conv_pattern.search(path))
    }

    original_linear_axis = {64, 128, 256, 512, 1024, 2048, 4096, 8192}
    extra_linear_axis = {1, 8, 32}
    expected_linear = {
        (input_features, output_features)
        for input_features in original_linear_axis | extra_linear_axis
        for output_features in original_linear_axis | extra_linear_axis
    }
    original_input_channels = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}
    original_output_channels = {1, 8, 16, 32, 64, 128, 256, 512}
    original_spatial_sizes = {32, 64, 128, 256, 512, 1024}
    original_configurations = {(3, 0), (3, 1), (5, 0), (5, 1)}
    max_spatial_size = {
        1: 1024,
        2: 1024,
        4: 1024,
        8: 1024,
        16: 512,
        32: 512,
        64: 256,
        128: 256,
        256: 128,
        512: 64,
    }
    original_conv = {
        (input_channels, image_size, kernel_size, padding, output_channels)
        for input_channels in original_input_channels
        for output_channels in original_output_channels
        for image_size in original_spatial_sizes
        if image_size <= max_spatial_size[input_channels]
        for kernel_size, padding in original_configurations
    }
    small_spatial_conv = {
        (input_channels, 2, 3, 1, output_channels)
        for input_channels in original_input_channels
        for output_channels in original_output_channels
    }
    pointwise_channels = {1, 8, 64, 512}
    pointwise_conv = {
        (input_channels, image_size, 1, 0, output_channels)
        for input_channels in pointwise_channels
        for output_channels in pointwise_channels
        for image_size in {2, 8, 32, 64}
    }
    expected_conv = original_conv | small_spatial_conv | pointwise_conv

    assert "121 Linear, 1648 Conv" in result.stdout
    assert len(model_paths) == len(set(model_paths)) == 1769
    assert len(expected_linear) - len(original_linear_axis) ** 2 == 57
    assert len(expected_conv) - len(original_conv) == 144
    assert linear_points == expected_linear
    assert conv_points == expected_conv


def test_pruned_validation_suite_schedules_dense_and_pruned_models(tmp_path):
    environment = os.environ.copy()
    environment.update(
        {
            "CONNECTION_CONFIG": str(PROJECT_ROOT / "measurement_hosts.example.json"),
            "OUTPUT_ROOT": str(tmp_path),
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(PROJECT_ROOT / "run_measurement_campaign.sh"),
            "--board",
            "dryrun",
            "--suite",
            "pruned_validation",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    model_paths = [
        match.group(1)
        for line in result.stdout.splitlines()
        if (match := re.match(r"^\[\d+/\d+\] (\S+)$", line))
    ]
    assert "2 PrunedValidation, 2 total" in result.stdout
    assert len(model_paths) == 2
    assert model_paths[0].endswith("ResNet18_32_10.pt")
    assert model_paths[1].endswith("PrunedResNet18_32_10.pt")