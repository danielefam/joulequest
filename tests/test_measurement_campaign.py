import os
import re
import subprocess
import sys
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


def test_pruned_validation_orin_suite_schedules_dense_and_orin_pruned_models(tmp_path):
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
            "agx_orin",
            "--suite",
            "pruned_validation_orin",
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
    assert "2 PrunedOrin, 0 PrunedPi5, 2 total" in result.stdout
    assert len(model_paths) == 2
    assert model_paths[0].endswith("ResNet18_32_10.pt")
    assert model_paths[1].endswith("PrunedOrinResNet18_32_10.pt")


def test_pruned_validation_pi5_suite_schedules_dense_and_pi5_pruned_models(tmp_path):
    environment = os.environ.copy()
    environment.update(
        {
            "CONNECTION_CONFIG": str(PROJECT_ROOT / "measurement_hosts.example.json"),
            "OUTPUT_ROOT": str(tmp_path),
            "BACKEND": "cpu",
            "MODEL_ROOT": "Models/CPU",
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(PROJECT_ROOT / "run_measurement_campaign.sh"),
            "--board",
            "pi5",
            "--suite",
            "pruned_validation_pi5",
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
    assert "0 PrunedOrin, 2 PrunedPi5, 2 total" in result.stdout
    assert len(model_paths) == 2
    assert model_paths[0] == "Models/CPU/ResNet18/ResNet18_32_10.pt"
    assert model_paths[1] == "Models/CPU/ResNet18/PrunedPi5ResNet18_32_10.pt"


def test_skips_already_completed_manifests(tmp_path):
    import json

    board_dir = tmp_path / "testboard"
    board_dir.mkdir(parents=True)
    manifest_payload = {
        "status": "COMPLETE",
        "model_path": "Models/CUDA/ResNet18/ResNet18_32_10.pt",
        "input_batch_size": 1,
    }
    (board_dir / "test_manifest.json").write_text(
        json.dumps(manifest_payload),
        encoding="utf-8",
    )

    fake_py = tmp_path / "fake_automated_measurement.py"
    fake_py.write_text(
        "#!/usr/bin/env python3\nimport sys\nprint('Fake measurement called for:', sys.argv)\nsys.exit(0)\n"
    )
    fake_py.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "CONNECTION_CONFIG": str(PROJECT_ROOT / "measurement_hosts.example.json"),
            "OUTPUT_ROOT": str(tmp_path),
        }
    )
    # We create a wrapper script or symlink automated_measurement.py in PROJECT_ROOT... wait, automated_measurement.py is invoked beside run_measurement_campaign.sh.
    # Instead, we can pass --continue-on-error and test with dry-run? Wait, dry-run prints commands without checking manifests.
    # But we can test manifest_exists_for_model directly by sourcing run_measurement_campaign.sh or running bash snippet!
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
            OUTPUT_DIRECTORY="{board_dir}"
            PYTHON_BIN="python"
            eval "$(sed -n '/^declare -A COMPLETED_MANIFESTS/,/^BOARD_LABEL=/p' "{PROJECT_ROOT}/run_measurement_campaign.sh" | sed 's/BOARD_LABEL=.*//')"
            manifest_exists_for_model "Models/CUDA/ResNet18/ResNet18_32_10.pt" 1 && echo "MATCH1"
            manifest_exists_for_model "Models/CUDA/ResNet18/PrunedResNet18_32_10.pt" 1 && echo "MATCH2" || echo "NOMATCH2"
            COMPLETED_MANIFESTS["Models/CUDA/ResNet18/PrunedResNet18_32_10.pt|1"]=1
            manifest_exists_for_model "Models/CUDA/ResNet18/PrunedResNet18_32_10.pt" 1 && echo "MATCH2_AFTER"
            """,
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "MATCH1" in result.stdout
    assert "NOMATCH2" in result.stdout
    assert "MATCH2_AFTER" in result.stdout


def test_validate_pruned_measurements_tool():
    import json

    result_pi5 = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "validate_pruned_measurements.py"),
            "--board",
            "pi5",
            "--json",
            "--predictions-only",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    pi5_data = json.loads(result_pi5.stdout)
    assert pi5_data["board"] == "pi5"
    assert round(pi5_data["predictions"]["pred_dense_energy_mJ"], 2) == 35.15
    assert round(pi5_data["predictions"]["pred_pruned_energy_mJ"], 2) == 7.92
    assert round(pi5_data["predictions"]["pred_savings_percent"], 1) == 77.5

    result_orin = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "validate_pruned_measurements.py"),
            "--board",
            "agx_orin",
            "--json",
            "--batch-size",
            "1",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    orin_data = json.loads(result_orin.stdout)
    assert orin_data["board"] == "agx_orin"
    assert round(orin_data["predictions"]["pred_dense_energy_mJ"], 2) == 18.57
    assert round(orin_data["predictions"]["pred_pruned_energy_mJ"], 2) == 10.82
    assert orin_data["measurements"]["dense_found"] is True
    assert orin_data["measurements"]["pruned_found"] is True
    assert orin_data["comparison"]["has_measurements"] is True