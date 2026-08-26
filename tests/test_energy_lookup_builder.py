from pathlib import Path

import pandas as pd

from processing_report.build_energy_lookup_table import (
    LOOKUP_COLUMNS,
    build_lookup_table,
    main,
)


def _accepted_row(model_path, energy):
    return {
        "model_path": model_path,
        "status": "COMPLETE",
        "quality_status": "OK",
        "energy_mean_mJ": energy,
        "detected_regions": 100,
        "expected_cycles": 100,
        "clock_alignment_status": "COMPLETE",
        "clock_alignment_uncertainty_s": 0.001,
        "clock_alignment_threshold_s": 0.005,
    }


def test_builder_keeps_only_strictly_accepted_measurements(tmp_path):
    base = _accepted_row("Models/CPU/Linear/Linear_2_4.pt", 3.0)
    rows = [
        base,
        {**base, "quality_status": "REVIEW", "energy_mean_mJ": 5.0},
        {**base, "energy_mean_mJ": -1.0},
        {**base, "detected_regions": 99},
        {**base, "clock_alignment_uncertainty_s": 0.01},
    ]
    summary = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)

    lookup = build_lookup_table([summary])

    assert tuple(lookup.columns) == LOOKUP_COLUMNS
    assert len(lookup) == 1
    assert lookup.iloc[0]["measurement_count"] == 2
    assert lookup.iloc[0]["energy_mean_mJ"] == 4.0


def test_builder_emits_current_linear_and_conv_schema(tmp_path):
    summary = tmp_path / "summary.csv"
    output = tmp_path / "energy_lookup_table.csv"
    pd.DataFrame(
        [
            _accepted_row("Models/CPU/Linear/Linear_64_128.pt", 1.25),
            _accepted_row("Models/CUDA/Conv/Conv_3_32_3_1_64.pt", 2.5),
            _accepted_row(
                "Models/CUDA/ResNet18/ResNetConv_64_128_16_3_2_1.pt", 3.5
            ),
        ]
    ).to_csv(summary, index=False)

    main([str(summary), "--output", str(output)])
    lookup = pd.read_csv(output)

    assert tuple(lookup.columns) == LOOKUP_COLUMNS
    assert set(lookup["layer_type"]) == {"linear", "conv"}
    assert set(lookup.loc[lookup["layer_type"] == "conv", "stride"]) == {1, 2}
    assert Path(output).is_file()