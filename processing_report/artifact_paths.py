"""Resolve measurement artifacts without loading processing dependencies."""

import json
from pathlib import Path


def get_manifest_file_path(measurements_manifest_dir_path, csv_path):
    manifest_directory = Path(measurements_manifest_dir_path)
    csv_stem = Path(csv_path).stem
    exact_matches = sorted(manifest_directory.rglob(f"{csv_stem}.json"))
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise RuntimeError(
            f"Ambiguous manifest match for {csv_path}: "
            + ", ".join(str(path) for path in exact_matches)
        )

    legacy_matches = sorted(
        path
        for path in manifest_directory.rglob("*.json")
        if csv_stem in path.stem
    )
    if len(legacy_matches) == 1:
        return legacy_matches[0]
    if len(legacy_matches) > 1:
        raise RuntimeError(
            f"Ambiguous legacy manifest match for {csv_path}: "
            + ", ".join(str(path) for path in legacy_matches)
        )
    return None


def load_measurement_metadata(manifest_dir, csv_path):
    manifest_path = get_manifest_file_path(manifest_dir, csv_path)
    if manifest_path is None:
        return None

    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    if manifest.get("status") != "COMPLETE":
        return None

    try:
        plan = manifest["plan"]
        acquisition = manifest["acquisition"]
        sampling_rate_hz = acquisition.get("achieved_sampling_rate_hz")
        if sampling_rate_hz is None:
            sampling_rate_hz = acquisition["sampling_rate_hz"]
        return {
            "campaign_id": manifest["campaign_id"],
            "model_path": manifest["model_path"],
            "status": manifest["status"],
            "quality_status": manifest.get("quality_status"),
            "inferences_per_cycle": plan["inferences_per_cycle"],
            "expected_cycles": plan["number_of_cycles"],
            "sampling_rate_hz": float(sampling_rate_hz),
        }
    except KeyError as error:
        raise KeyError(
            f"Manifest {manifest_path} lacks required processing metadata"
        ) from error