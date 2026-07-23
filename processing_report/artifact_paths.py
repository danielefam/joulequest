"""Resolve measurement artifacts without loading processing dependencies."""

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