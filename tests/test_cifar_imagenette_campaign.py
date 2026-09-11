import os
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cifar_imagenette_campaign_dry_run(tmp_path):
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
            str(PROJECT_ROOT / "run_cifar_imagenette_campaign.sh"),
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "JOULEQUEST FOCUSED CAMPAIGN: CIFAR-10 & IMAGENETTE RESNET-18 (BATCH 1)" in result.stdout
    assert "Total models: 468" in result.stdout
    assert "Scheduled: 468; completed: 0; skipped: 0; failed: 0." in result.stdout

    # Check batch size argument in scheduled commands defaults to 1
    assert "--batch-size 1" in result.stdout

    # Verify that unneeded architectures are NOT scheduled
    assert "Attention_" not in result.stdout
    assert "RotaryAttention_" not in result.stdout
    assert "Lenet" not in result.stdout
    assert "MaxPool_8_28_2" not in result.stdout
    assert "Linear_8192" not in result.stdout

    # Verify key ResNet CIFAR and Imagenette layers are scheduled
    assert "Models/CUDA/Conv/Conv_3_32_3_1_64.pt" in result.stdout
    assert "Models/CUDA/ResNet18/ResNetConv_3_64_128_7_2_3.pt" in result.stdout
    assert "Models/CUDA/ResNet18/ResNetConv_64_128_16_3_2_1.pt" in result.stdout
    assert "Models/CUDA/ResNet18/ResNet18_32_10.pt" not in result.stdout
    assert "Models/CUDA/Linear/Linear_512_10.pt" in result.stdout


def test_cifar_imagenette_campaign_custom_batch_size(tmp_path):
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
            str(PROJECT_ROOT / "run_cifar_imagenette_campaign.sh"),
            "--board",
            "test_orin_bs32",
            "--batch-size",
            "32",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "JOULEQUEST FOCUSED CAMPAIGN: CIFAR-10 & IMAGENETTE RESNET-18 (BATCH 32)" in result.stdout
    assert "--batch-size 32" in result.stdout

