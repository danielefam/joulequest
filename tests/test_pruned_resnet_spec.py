import copy
import json
from pathlib import Path

import pytest

from layers.pruned_resnet import PrunedResNet18Spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRUNING_LOG = PROJECT_ROOT / "measurements" / "pruning_logs" / (
    "resnet18_icpr_weight_1p2_lookup_agxorinbs32_energy_lookup_"
    "out_of_range_extrapolate_seed_0_energy_mode_discrete_998752_pruning.json"
)


def test_example_log_preserves_explicit_shortcuts_and_operation_schedule():
    spec = PrunedResNet18Spec.from_pruning_log(PRUNING_LOG)

    assert spec.architecture_id == "44450f2dfeb2c122"
    assert [block.name for block in spec.blocks if block.shortcut] == [
        "layer2.0",
        "layer3.0",
        "layer4.0",
    ]
    assert sum(operation.count for operation in spec.estimator_operations()) == 21
    assert {operation.model_name for operation in spec.estimator_operations()} >= {
        "ResNetConv_62_124_16_1_2_0.pt",
        "ResNetConv_128_105_8_1_2_0.pt",
        "ResNetConv_88_512_4_1_2_0.pt",
        "Linear_255_10.pt",
    }


def test_runtime_model_uses_only_logged_projection_convolutions():
    torch = pytest.importorskip("torch")
    from layers.resnet import PrunedCifarResNet18, PrunedResNetBasicBlock

    spec = PrunedResNet18Spec.from_pruning_log(PRUNING_LOG)
    model = PrunedCifarResNet18(config=spec.to_model_config())

    blocks = [block for stage in (model.layer1, model.layer2, model.layer3, model.layer4) for block in stage]
    assert all(isinstance(block, PrunedResNetBasicBlock) for block in blocks)
    assert sum(block.shortcut is not None for block in blocks) == 3
    assert model.layer1[0].shortcut is None
    assert model.layer1[1].shortcut is None

    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 10)


def test_invalid_main_path_continuity_is_rejected():
    data = json.loads(PRUNING_LOG.read_text(encoding="utf-8"))
    broken = copy.deepcopy(data)
    broken["layer1.1.conv1"]["active_input"] = 57

    with pytest.raises(ValueError, match="previous output"):
        PrunedResNet18Spec.from_pruning_data(broken)