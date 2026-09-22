import csv
from pathlib import Path

from layers.pruned_resnet import PrunedResNet18Spec
from processing_report.validate_pruned_measurements import (
    compose_estimator_scope_energy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRUNING_LOG = PROJECT_ROOT / "measurements" / "pruning_logs" / (
    "resnet18_icpr_weight_1p2_lookup_agxorinbs32_energy_lookup_"
    "out_of_range_extrapolate_seed_0_energy_mode_discrete_998752_pruning.json"
)


def _write_summary(path, rows):
    with path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_composition_weights_exact_architecture_components(tmp_path):
    spec = PrunedResNet18Spec.from_pruning_log(PRUNING_LOG)
    rows = []
    for index, operation in enumerate(spec.estimator_operations()):
        rows.append(
            {
                "campaign_id": f"20260922T0000{index:02d}Z_component_bs32_run",
                "model_path": operation.model_name,
                "architecture_id": spec.architecture_id,
                "status": "COMPLETE",
                "quality_status": "OK",
                "energy_mean_mJ": "2.0",
            }
        )
    rows.append(
        {
            "campaign_id": "20260922T010000Z_pruned_bs32_run",
            "model_path": "PrunedResNet18_32_10.pt",
            "architecture_id": spec.architecture_id,
            "status": "COMPLETE",
            "quality_status": "OK",
            "energy_mean_mJ": "50.0",
        }
    )
    summary = tmp_path / "summary.csv"
    _write_summary(summary, rows)

    result = compose_estimator_scope_energy(summary, PRUNING_LOG, batch_size=32)

    assert result["complete"] is True
    assert result["expected_operation_calls"] == 21
    assert result["composed_energy_mJ"] == 42.0
    assert result["residual_energy_mJ"] == 8.0
    assert result["residual_fraction"] == 0.16


def test_composition_does_not_publish_partial_total(tmp_path):
    spec = PrunedResNet18Spec.from_pruning_log(PRUNING_LOG)
    summary = tmp_path / "summary.csv"
    _write_summary(
        summary,
        [
            {
                "campaign_id": "20260922T010000Z_pruned_bs32_run",
                "model_path": "PrunedResNet18_32_10.pt",
                "architecture_id": spec.architecture_id,
                "status": "COMPLETE",
                "quality_status": "OK",
                "energy_mean_mJ": "50.0",
            }
        ],
    )

    result = compose_estimator_scope_energy(summary, PRUNING_LOG, batch_size=32)

    assert result["complete"] is False
    assert result["composed_energy_mJ"] is None
    assert len(result["missing_components"]) == 21