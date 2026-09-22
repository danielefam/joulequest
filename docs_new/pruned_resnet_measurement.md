# Pruned ResNet-18 composition measurement

**Document date:** 2026-09-22  
**Architecture schema:** 1

## Purpose

This workflow tests whether the sum of separately measured pruned ResNet-18
`Conv2d` and `Linear` operations agrees with the physical INA226 measurement of
the complete pruned network. One JouleNAS pruning JSON is the source of truth
for model construction, component scheduling, identity, and reporting.

$$
E_{estimator}=\sum_i n_i E_{Conv2d,i}+E_{Linear}
$$

The report also computes $E_{residual}=E_{whole}-E_{estimator}$. The residual
includes BatchNorm, activations, pooling, residual addition, kernel fusion,
launch costs, and runtime overhead. It is not automatically measurement error.

## Run the experiment

The dedicated wrapper was removed. Use the general launcher:

```bash
BACKEND=cuda NETWORK_BATCH_SIZE=32 \
./run_measurement_campaign.sh \
  --board agx_orin_bs32 \
  --suite pruned_resnet18 \
  --pruning-log measurements/pruning_logs/resnet18_icpr_weight_1p2_lookup_agxorinbs32_energy_lookup_out_of_range_extrapolate_seed_0_energy_mode_discrete_998752_pruning.json
```

Use `--dry-run` first. For the example log, the suite creates 23 experiments:
one dense model, one runtime-configured pruned model, and 21 estimator-scope
coordinates. Every run uses batch 32 and architecture ID `44450f2dfeb2c122`.

The pruning path is local to the controller. Canonical JSON crosses SSH, so
controller and runner do not need matching filesystem paths.

## Process and compare

Process the campaign normally with `processing_report/process_and_visualize.py`.
Summary rows now include `architecture_id`. Then run:

```bash
python processing_report/validate_pruned_measurements.py \
  --board agx_orin_bs32 \
  --batch-size 32 \
  --summary-csv measurements/lookup_summaries/summaries/agx_orin_bs32.csv \
  --pruning-log measurements/pruning_logs/resnet18_icpr_weight_1p2_lookup_agxorinbs32_energy_lookup_out_of_range_extrapolate_seed_0_energy_mode_discrete_998752_pruning.json \
  --composition-only
```

Add `--json` for machine-readable output. The report stays incomplete when a
required coordinate or matching whole-model measurement is missing.

## Shortcut semantics

Only a logged `downsample.0` creates a learned 1x1 convolution. The example has
three: `layer2.0`, `layer3.0`, and `layer4.0`. An identity shortcut with changed
channels slices excess channels or appends zeros, avoiding an unlogged and
unestimated convolution.

## Code index

| Responsibility | Code owner | Main symbols |
| --- | --- | --- |
| Parse and validate logs | `layers/pruned_resnet.py` | `PrunedResNet18Spec.from_pruning_log`, `from_pruning_data` |
| Identity and SSH payload | `layers/pruned_resnet.py` | `canonical_json`, `architecture_id` |
| Generate component coordinates | `layers/pruned_resnet.py` | `estimator_operations` |
| Print campaign plan | `plan_pruned_measurement.py` | `main` |
| Select and run suite | `run_measurement_campaign.sh` | `pruned_resnet18`, `run_experiment` |
| Identity shortcut alignment | `layers/resnet.py` | `PrunedResNetBasicBlock._identity_shortcut` |
| Runtime configured model | `runner.py` | `TorchRunner._build_model` |
| SSH argument transport | `automated_measurement.py` | `RemoteExperimentController._remote_arguments` |
| Manifest identity | `run_manager.py` | `RunManager._new_campaign_id`, `execute` |
| Summary identity | `processing_report/artifact_paths.py`, `processing_report/process_and_visualize.py` | `load_measurement_metadata`, `process_file` |
| Weighted comparison | `processing_report/validate_pruned_measurements.py` | `compose_estimator_scope_energy`, `format_composition_report` |
| Spec/model tests | `tests/test_pruned_resnet_spec.py` | example-log tests |
| Campaign tests | `tests/test_measurement_campaign.py` | runtime suite test |
| Report tests | `tests/test_pruned_measurement_validation.py` | complete/incomplete totals |

## Compatibility

Named pruned classes and `pruned_validation_*` aliases remain for historical
campaigns. New work should use `pruned_resnet18`. `materialize_pruned_model.py`
is inspection-only; `--update-resnet` is rejected.