#!/usr/bin/env bash
set -euo pipefail

# Map MMLSv2 to the Mars-LS representation and compare it with MARS_LSC_ROOT.
# The three roots are loaded from the repository .env through _env.sh:
#   MMLSV2_SRC_DATA_ROOT  source MMLSv2 root
#   MMLSV2_DATA_ROOT      mapped output root
#   MARS_LSC_ROOT         reference Mars-LS root
#
# Optional overrides:
#   PYTHON_BIN=python
#   SPLITS="train val test"
#   ATOL=0.001
#   OVERWRITE=1
#   IGNORE_MASKS=1

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

. "${SCRIPT_DIR}/_env.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLITS="${SPLITS:-train val test}"
ATOL="${ATOL:-0.001}"
OVERWRITE="${OVERWRITE:-0}"
IGNORE_MASKS="${IGNORE_MASKS:-0}"

# OpenCV emits warnings for GeoTIFF metadata tags that it does not interpret.
# They do not affect the pixel data used by this mapping workflow.
export OPENCV_LOG_LEVEL=ERROR

: "${MMLSV2_SRC_DATA_ROOT:?MMLSV2_SRC_DATA_ROOT is not set in .env}"
: "${MMLSV2_DATA_ROOT:?MMLSV2_DATA_ROOT is not set in .env}"
: "${MARS_LSC_ROOT:?MARS_LSC_ROOT is not set in .env}"

if [ ! -d "${MMLSV2_SRC_DATA_ROOT}" ]; then
  echo "Source dataset does not exist: ${MMLSV2_SRC_DATA_ROOT}" >&2
  exit 1
fi
if [ ! -d "${MARS_LSC_ROOT}" ]; then
  echo "Reference dataset does not exist: ${MARS_LSC_ROOT}" >&2
  exit 1
fi
if [ "${MMLSV2_SRC_DATA_ROOT}" = "${MMLSV2_DATA_ROOT}" ]; then
  echo "MMLSV2_DATA_ROOT must differ from MMLSV2_SRC_DATA_ROOT" >&2
  exit 1
fi
if [ "${MARS_LSC_ROOT}" = "${MMLSV2_DATA_ROOT}" ]; then
  echo "MMLSV2_DATA_ROOT must differ from MARS_LSC_ROOT" >&2
  exit 1
fi

read -r -a SPLIT_LIST <<< "${SPLITS}"
if [ "${#SPLIT_LIST[@]}" -eq 0 ]; then
  echo "SPLITS must contain at least one split" >&2
  exit 1
fi

MAP_ARGS=(
  --source-root "${MMLSV2_SRC_DATA_ROOT}"
  --output-root "${MMLSV2_DATA_ROOT}"
  --reference-root "${MARS_LSC_ROOT}"
  --splits "${SPLIT_LIST[@]}"
  --test-masks
)
if [ "${OVERWRITE}" = "1" ] || [ "${OVERWRITE}" = "true" ] || [ "${OVERWRITE}" = "TRUE" ]; then
  MAP_ARGS+=(--overwrite)
fi

echo "[map] source=${MMLSV2_SRC_DATA_ROOT}"
echo "[map] output=${MMLSV2_DATA_ROOT}"
echo "[map] reference=${MARS_LSC_ROOT}"
echo "[map] splits=${SPLITS} overwrite=${OVERWRITE}"

"${PYTHON_BIN}" tools/data/map_mmlsv2_to_mars.py "${MAP_ARGS[@]}"

# The mapping tool above performs the conversion.  The following small Python
# block compares the materialized output with the reference TIFF files.
"${PYTHON_BIN}" - "${MMLSV2_DATA_ROOT}" "${MARS_LSC_ROOT}" "${ATOL}" "${IGNORE_MASKS}" "${SPLIT_LIST[@]}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm


mapped_root = Path(sys.argv[1]).resolve()
reference_root = Path(sys.argv[2]).resolve()
atol = float(sys.argv[3])
ignore_masks = sys.argv[4].strip().lower() in {"1", "true", "yes", "on"}
splits = sys.argv[5:]

if atol < 0:
    raise SystemExit("ATOL must be non-negative")


def tif_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    result = {}
    for pattern in ("*.tif", "*.tiff"):
        for path in directory.glob(pattern):
            result[path.name] = path
    return result


def compare_file_set(mapped_dir: Path, reference_dir: Path, tolerance: float):
    mapped = tif_files(mapped_dir)
    reference = tif_files(reference_dir)
    mapped_names = set(mapped)
    reference_names = set(reference)
    common_names = sorted(mapped_names & reference_names)
    exact_count = 0
    close_count = 0
    shape_mismatch = 0
    mean_errors = []
    max_errors = []
    mismatches = []

    for name in tqdm(
        common_names,
        desc=f"Compare {mapped_dir.parent.name}/{mapped_dir.name}",
        unit="file",
    ):
        left = np.asarray(tifffile.imread(str(mapped[name])))
        right = np.asarray(tifffile.imread(str(reference[name])))
        if left.shape != right.shape:
            shape_mismatch += 1
            mismatches.append({
                "name": name,
                "shape_left": list(left.shape),
                "shape_right": list(right.shape),
            })
            continue

        diff = np.abs(left.astype(np.float64) - right.astype(np.float64))
        exact = bool(np.all(diff == 0))
        close = bool(np.all(diff <= tolerance))
        exact_count += int(exact)
        close_count += int(close)
        mean_errors.append(float(np.mean(diff)))
        max_errors.append(float(np.max(diff)))
        if not close:
            mismatches.append({
                "name": name,
                "shape": list(left.shape),
                "dtype_left": str(left.dtype),
                "dtype_right": str(right.dtype),
                "exact_fraction": float(np.mean(diff == 0)),
                "within_atol_fraction": float(np.mean(diff <= tolerance)),
                "mean_absolute_error": float(np.mean(diff)),
                "max_absolute_error": float(np.max(diff)),
            })

    report = {
        "mapped_count": len(mapped_names),
        "reference_count": len(reference_names),
        "common_count": len(common_names),
        "missing_in_mapped": sorted(reference_names - mapped_names),
        "extra_in_mapped": sorted(mapped_names - reference_names),
        "compared_count": len(common_names),
        "exact_match_count": exact_count,
        "within_atol_count": close_count,
        "shape_mismatch_count": shape_mismatch,
        "exact_match": (
            mapped_names == reference_names
            and exact_count == len(common_names)
            and shape_mismatch == 0
        ),
        "within_atol_match": (
            mapped_names == reference_names
            and close_count == len(common_names)
            and shape_mismatch == 0
        ),
        "mean_absolute_error_per_file": (
            float(np.mean(mean_errors)) if mean_errors else None
        ),
        "max_absolute_error": max(max_errors) if max_errors else None,
        "mismatches": mismatches,
    }
    return report


split_reports = {}
for split in splits:
    image_report = compare_file_set(
        mapped_root / split / "images",
        reference_root / split / "images",
        atol,
    )
    split_report = {
        "images": image_report,
        "images_match": bool(image_report["within_atol_match"]),
    }

    if ignore_masks:
        split_report["masks"] = {"status": "ignored"}
        split_report["masks_match"] = None
    else:
        mapped_masks = tif_files(mapped_root / split / "masks")
        reference_masks = tif_files(reference_root / split / "masks")
        if not mapped_masks and not reference_masks:
            split_report["masks"] = {"status": "not_available"}
            split_report["masks_match"] = None
        elif not reference_masks:
            # Competition test labels may be hidden/unavailable while MMLSv2
            # provides public masks.  Do not treat those extra public labels
            # as an image mapping failure.
            split_report["masks"] = {
                "status": "reference_not_available",
                "mapped_count": len(mapped_masks),
                "reference_count": 0,
            }
            split_report["masks_match"] = None
        else:
            mask_report = compare_file_set(
                mapped_root / split / "masks",
                reference_root / split / "masks",
                0.0,
            )
            split_report["masks"] = {
                "status": "checked",
                "files": mask_report,
            }
            split_report["masks_match"] = bool(mask_report["exact_match"])

    split_report["complete_match"] = bool(
        split_report["images_match"]
        and split_report["masks_match"] is not False
    )
    split_reports[split] = split_report

image_complete = all(item["images_match"] for item in split_reports.values())
checked_masks = [
    item["masks_match"]
    for item in split_reports.values()
    if item["masks_match"] is not None
]
masks_complete = all(checked_masks) if checked_masks else None
complete = bool(image_complete and masks_complete is not False)

report = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "mapped_root": str(mapped_root),
    "reference_root": str(reference_root),
    "splits": splits,
    "atol": atol,
    "ignore_masks": ignore_masks,
    "split_reports": split_reports,
    "images_complete_match": image_complete,
    "masks_complete_match": masks_complete,
    "complete_match": complete,
}
report_path = mapped_root / "mapping_match_report.json"
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)

print("=== MMLSv2 mapping comparison ===")
for split, item in split_reports.items():
    images = item["images"]
    print(
        f"[{split}] images: mapped={images['mapped_count']} "
        f"reference={images['reference_count']} "
        f"match={item['images_match']}"
    )
    masks = item["masks"]
    if masks["status"] == "checked":
        print(f"[{split}] masks : match={item['masks_match']}")
    else:
        print(f"[{split}] masks : {masks['status']}")
    print(f"[{split}] complete: {item['complete_match']}")
print(f"Images completely match: {image_complete}")
print(f"Masks completely match : {masks_complete}")
print(f"Dataset completely match: {complete}")
print(f"Report: {report_path}")

raise SystemExit(0 if complete else 2)
PY
