from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from netCDF4 import Dataset


LABEL_MAP = {
    0: "Clear",
    1: "Water Type",
    2: "Super Cooled Type",
    3: "Mixed Type",
    4: "Ice Type",
}

EXCLUDED_LABELS = {
    5: "Uncertain",
    126: "Space",
    127: "FillValue",
}

# 动态步长：训练集重叠采样；测试集固定 stride=64，保持评估干净
TRAIN_STRIDE_MAP = {
    0: 128,
    1: 64,
    2: 32,
    3: 16,
    4: 64,
}
TEST_STRIDE = 64

# 重新按“月份约束”划分：
# - train/test 都包含 01 / 04 / 07 / 10 月
# - test 每个月固定 4 景，共 16 景
TEST_SCENES = [
    "20250402000000",
    "20250412120000",
    "20250421060000",
    "20250421180000",
    "20250706060000",
    "20250713180000",
    "20250722060000",
    "20250730180000",
    "20251004120000",
    "20251012060000",
    "20251012120000",
    "20251028180000",
    "20260103000000",
    "20260111120000",
    "20260122000000",
    "20260122180000",
]

FDI_PATTERN = re.compile(r"FY4B-.*?_L1-_FDI-_MULT_NOM_(\d{14})_(\d{14})_4000M_.*\.HDF$")
CLP_PATTERN = re.compile(r"FY4B-.*?_L2-_CLP-_MULT_NOM_(\d{14})_(\d{14})_4000M_.*\.NC$")


@dataclass
class ScenePair:
    start_time: str
    end_time: str
    fdi_path: Path
    clp_path: Path


@dataclass
class ScenePlan:
    split: str
    pair: ScenePair
    rows: np.ndarray
    cols: np.ndarray
    labels: np.ndarray
    purities: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.labels.size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FY4B ViT dataset with month-stratified scene split.")
    parser.add_argument("--input-dir", type=Path, default=Path("data_raw"), help="Directory with FY4B HDF/NC files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_processed/fy4b_vit_month_stratified_scene_split"),
        help="Independent output directory.",
    )
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--purity-threshold", type=float, default=0.80)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def build_scene_pairs(input_dir: Path) -> list[ScenePair]:
    fdi_by_start: dict[str, tuple[str, Path]] = {}
    clp_by_start: dict[str, tuple[str, Path]] = {}

    for path in sorted(input_dir.glob("*.HDF")):
        m = FDI_PATTERN.match(path.name)
        if m:
            fdi_by_start[m.group(1)] = (m.group(2), path)

    for path in sorted(input_dir.glob("*.NC")):
        m = CLP_PATTERN.match(path.name)
        if m:
            clp_by_start[m.group(1)] = (m.group(2), path)

    common = sorted(set(fdi_by_start) & set(clp_by_start))
    if not common:
        raise FileNotFoundError("No matched FY4B FDI/CLP pairs found.")

    missing_fdi = sorted(set(clp_by_start) - set(fdi_by_start))
    missing_clp = sorted(set(fdi_by_start) - set(clp_by_start))
    if missing_fdi or missing_clp:
        raise RuntimeError(f"Pair mismatch. missing_fdi={missing_fdi[:5]}, missing_clp={missing_clp[:5]}")

    pairs: list[ScenePair] = []
    for key in common:
        fdi_end, fdi_path = fdi_by_start[key]
        clp_end, clp_path = clp_by_start[key]
        if fdi_end != clp_end:
            raise RuntimeError(f"Time mismatch for {key}: {fdi_end} vs {clp_end}")
        pairs.append(ScenePair(key, fdi_end, fdi_path, clp_path))
    return pairs


def load_clp_array(clp_path: Path) -> np.ndarray:
    with Dataset(clp_path) as ds:
        clp_var = ds.variables["CLP"]
        arr = clp_var[:]
        fill_value = getattr(clp_var, "FillValue", 127)
        if np.ma.isMaskedArray(arr):
            arr = arr.filled(fill_value)
        else:
            arr = np.asarray(arr)
    return np.asarray(arr)


def make_integral(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int32)
    out[1:, 1:] = mask.astype(np.int32, copy=False).cumsum(axis=0).cumsum(axis=1)
    return out


def query_integral(integral: np.ndarray, rows: np.ndarray, cols: np.ndarray, patch_size: int) -> np.ndarray:
    bottom = rows + patch_size
    right = cols + patch_size
    return integral[bottom, right] - integral[rows, right] - integral[bottom, cols] + integral[rows, cols]


def build_invalid_fdi_mask(hdf_path: Path) -> np.ndarray:
    invalid_any: np.ndarray | None = None
    with h5py.File(hdf_path, "r") as hdf:
        for channel_idx in range(1, 16):
            ds = hdf[f"Data/NOMChannel{channel_idx:02d}"]
            fill_value = int(np.asarray(ds.attrs["FillValue"]).reshape(-1)[0])
            cur = ds[:] == fill_value
            invalid_any = cur if invalid_any is None else (invalid_any | cur)
    if invalid_any is None:
        raise RuntimeError(f"No FDI channels found in {hdf_path.name}")
    return invalid_any


def plan_scene(
    split: str,
    pair: ScenePair,
    clp_array: np.ndarray,
    invalid_fdi_mask: np.ndarray,
    patch_size: int,
    purity_threshold: float,
    stride_map: dict[int, int],
) -> ScenePlan | None:
    allowed_labels = np.array(sorted(LABEL_MAP), dtype=np.int16)
    invalid_clp_mask = ~np.isin(clp_array, allowed_labels)
    invalid_clp_integral = make_integral(invalid_clp_mask)
    invalid_fdi_integral = make_integral(invalid_fdi_mask)
    class_integrals = {label: make_integral(clp_array == label) for label in allowed_labels.tolist()}

    max_row = clp_array.shape[0] - patch_size
    max_col = clp_array.shape[1] - patch_size
    if max_row < 0 or max_col < 0:
        return None

    need_pixels = math.ceil(purity_threshold * patch_size * patch_size)
    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    purities_all: list[np.ndarray] = []

    for stride in sorted(set(stride_map.values())):
        row_starts = np.arange(0, max_row + 1, stride, dtype=np.int32)
        col_starts = np.arange(0, max_col + 1, stride, dtype=np.int32)
        rr, cc = np.meshgrid(row_starts, col_starts, indexing="ij")
        rows = rr.reshape(-1)
        cols = cc.reshape(-1)

        valid_mask = query_integral(invalid_clp_integral, rows, cols, patch_size) == 0
        if not np.any(valid_mask):
            continue
        rows = rows[valid_mask]
        cols = cols[valid_mask]

        label_counts = np.stack(
            [query_integral(class_integrals[label], rows, cols, patch_size) for label in allowed_labels],
            axis=1,
        )
        dominant_idx = label_counts.argmax(axis=1)
        dominant_counts = label_counts.max(axis=1)
        dominant_labels = allowed_labels[dominant_idx]

        purity_mask = dominant_counts >= need_pixels
        stride_label_mask = np.array([stride_map[int(x)] == stride for x in dominant_labels], dtype=bool)
        keep = purity_mask & stride_label_mask
        if not np.any(keep):
            continue

        rows = rows[keep]
        cols = cols[keep]
        dominant_labels = dominant_labels[keep].astype(np.uint8)
        dominant_counts = dominant_counts[keep]

        valid_fdi = query_integral(invalid_fdi_integral, rows, cols, patch_size) == 0
        if not np.any(valid_fdi):
            continue

        rows_all.append(rows[valid_fdi])
        cols_all.append(cols[valid_fdi])
        labels_all.append(dominant_labels[valid_fdi])
        purities_all.append((dominant_counts[valid_fdi] / float(patch_size * patch_size)).astype(np.float32))

    if not rows_all:
        return None

    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    labels = np.concatenate(labels_all)
    purities = np.concatenate(purities_all)
    order = np.lexsort((cols, rows))
    return ScenePlan(split, pair, rows[order], cols[order], labels[order], purities[order])


def iter_patch_batches(raw_array: np.ndarray, rows: np.ndarray, cols: np.ndarray, patch_size: int, batch_size: int) -> Iterable[np.ndarray]:
    total = rows.size
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = np.stack([raw_array[r:r + patch_size, c:c + patch_size] for r, c in zip(rows[start:end], cols[start:end])], axis=0)
        yield batch


def compute_train_stats(scene_plans: list[ScenePlan], patch_size: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    pixel_sum = np.zeros(15, dtype=np.float64)
    pixel_sumsq = np.zeros(15, dtype=np.float64)
    total_pixels = sum(plan.sample_count for plan in scene_plans) * patch_size * patch_size

    for idx, plan in enumerate(scene_plans, start=1):
        print(f"[train-stats {idx:02d}/{len(scene_plans):02d}] {plan.pair.start_time} -> {plan.sample_count} samples")
        with h5py.File(plan.pair.fdi_path, "r") as hdf:
            for channel_idx in range(1, 16):
                raw = hdf[f"Data/NOMChannel{channel_idx:02d}"][:]
                lut = np.asarray(hdf[f"Calibration/CALChannel{channel_idx:02d}"][:], dtype=np.float32)
                for batch in iter_patch_batches(raw, plan.rows, plan.cols, patch_size, batch_size):
                    calibrated = lut[batch]
                    pixel_sum[channel_idx - 1] += calibrated.sum(dtype=np.float64)
                    cal64 = calibrated.astype(np.float64, copy=False)
                    pixel_sumsq[channel_idx - 1] += np.square(cal64).sum(dtype=np.float64)

    mean = pixel_sum / total_pixels
    var = np.maximum(pixel_sumsq / total_pixels - np.square(mean), 1e-12)
    std = np.sqrt(var)
    return mean, std


def write_split(
    split_name: str,
    scene_plans: list[ScenePlan],
    output_dir: Path,
    patch_size: int,
    batch_size: int,
    out_dtype: np.dtype,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[tuple[int, int, int, int], tuple[int], Counter[int]]:
    total_samples = sum(plan.sample_count for plan in scene_plans)
    x_shape = (total_samples, 15, patch_size, patch_size)
    y_shape = (total_samples,)

    x_memmap = np.lib.format.open_memmap(output_dir / f"{split_name}_x_fy4b.npy", mode="w+", dtype=out_dtype, shape=x_shape)
    y_memmap = np.lib.format.open_memmap(output_dir / f"{split_name}_y_fy4b.npy", mode="w+", dtype=np.uint8, shape=y_shape)
    label_counter: Counter[int] = Counter()

    with (output_dir / f"{split_name}_patch_metadata.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "split", "start_time", "end_time", "month", "fdi_file", "clp_file", "row_start", "col_start", "label", "label_name", "purity"])

        offset = 0
        for idx, plan in enumerate(scene_plans, start=1):
            print(f"[write-{split_name} {idx:02d}/{len(scene_plans):02d}] {plan.pair.start_time} -> {plan.sample_count} samples")
            with h5py.File(plan.pair.fdi_path, "r") as hdf:
                for channel_idx in range(1, 16):
                    raw = hdf[f"Data/NOMChannel{channel_idx:02d}"][:]
                    lut = np.asarray(hdf[f"Calibration/CALChannel{channel_idx:02d}"][:], dtype=np.float32)
                    write_pos = offset
                    for batch in iter_patch_batches(raw, plan.rows, plan.cols, patch_size, batch_size):
                        calibrated = lut[batch]
                        normalized = (calibrated - mean[channel_idx - 1]) / std[channel_idx - 1]
                        batch_len = normalized.shape[0]
                        x_memmap[write_pos:write_pos + batch_len, channel_idx - 1] = normalized.astype(out_dtype, copy=False)
                        write_pos += batch_len

            y_memmap[offset:offset + plan.sample_count] = plan.labels
            label_counter.update(int(x) for x in plan.labels.tolist())

            for j in range(plan.sample_count):
                w.writerow([
                    offset + j,
                    split_name,
                    plan.pair.start_time,
                    plan.pair.end_time,
                    plan.pair.start_time[4:6],
                    plan.pair.fdi_path.name,
                    plan.pair.clp_path.name,
                    int(plan.rows[j]),
                    int(plan.cols[j]),
                    int(plan.labels[j]),
                    LABEL_MAP[int(plan.labels[j])],
                    float(plan.purities[j]),
                ])
            offset += plan.sample_count

    x_memmap.flush()
    y_memmap.flush()
    del x_memmap
    del y_memmap
    return x_shape, y_shape, label_counter


def write_scene_manifest(split_name: str, plans: list[ScenePlan], output_dir: Path) -> None:
    with (output_dir / f"{split_name}_scene_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["split", "start_time", "end_time", "month", "fdi_file", "clp_file", "samples_kept"])
        for plan in plans:
            w.writerow([
                split_name,
                plan.pair.start_time,
                plan.pair.end_time,
                plan.pair.start_time[4:6],
                plan.pair.fdi_path.name,
                plan.pair.clp_path.name,
                plan.sample_count,
            ])


def write_readme(output_dir: Path, summary: dict) -> None:
    content = f"""# FY4B ViT Dataset

## 规则
- 先按景划分 train/test，再切片，避免泄漏
- train/test 都覆盖 01 / 04 / 07 / 10 月
- 训练集使用动态步长重叠采样
- 测试集固定 stride=64，不做重叠增强

## 切片参数
- patch size: {summary["patch_size"]}x{summary["patch_size"]}
- purity threshold: {summary["purity_threshold"]}
- train stride map: {json.dumps(summary["train_stride_map"], ensure_ascii=False)}
- test stride: {summary["test_stride"]}

## 文件
- train_x_fy4b.npy / train_y_fy4b.npy
- test_x_fy4b.npy / test_y_fy4b.npy
- train_patch_metadata.csv / test_patch_metadata.csv
- train_scene_manifest.csv / test_scene_manifest.csv
- channel_stats_train.npz
- scene_split.json
- dataset_summary.json
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_pairs = build_scene_pairs(input_dir)
    all_scenes = {pair.start_time for pair in scene_pairs}
    test_scene_set = set(TEST_SCENES)
    train_scene_set = all_scenes - test_scene_set
    if not test_scene_set.issubset(all_scenes):
        missing = sorted(test_scene_set - all_scenes)
        raise RuntimeError(f"Configured test scenes missing: {missing}")

    print(f"Matched {len(scene_pairs)} FY4B pairs.")
    print(f"Train scenes: {len(train_scene_set)} | Test scenes: {len(test_scene_set)}")

    train_plans: list[ScenePlan] = []
    test_plans: list[ScenePlan] = []

    for idx, pair in enumerate(scene_pairs, start=1):
        split = "test" if pair.start_time in test_scene_set else "train"
        stride_map = {label: TEST_STRIDE for label in LABEL_MAP} if split == "test" else TRAIN_STRIDE_MAP
        print(f"[plan {idx:02d}/{len(scene_pairs):02d}] {pair.start_time} -> {split}")
        clp_array = load_clp_array(pair.clp_path)
        invalid_fdi_mask = build_invalid_fdi_mask(pair.fdi_path)
        plan = plan_scene(split, pair, clp_array, invalid_fdi_mask, args.patch_size, args.purity_threshold, stride_map)
        if plan is None or plan.sample_count == 0:
            print("  kept 0")
            continue
        print(f"  kept {plan.sample_count} samples")
        (test_plans if split == "test" else train_plans).append(plan)

    if not train_plans or not test_plans:
        raise RuntimeError("Train or test split is empty after filtering.")

    print("Computing train-only normalization statistics...")
    mean, std = compute_train_stats(train_plans, args.patch_size, args.batch_size)
    out_dtype = np.float16 if args.dtype == "float16" else np.float32

    print("Writing train arrays...")
    train_x_shape, train_y_shape, train_counts = write_split(
        "train", train_plans, output_dir, args.patch_size, args.batch_size, out_dtype, mean, std
    )
    print("Writing test arrays...")
    test_x_shape, test_y_shape, test_counts = write_split(
        "test", test_plans, output_dir, args.patch_size, args.batch_size, out_dtype, mean, std
    )

    write_scene_manifest("train", train_plans, output_dir)
    write_scene_manifest("test", test_plans, output_dir)

    np.savez(output_dir / "channel_stats_train.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))

    scene_split = {
        "train_scenes": sorted(train_scene_set),
        "test_scenes": sorted(test_scene_set),
        "train_months": sorted({scene[4:6] for scene in train_scene_set}),
        "test_months": sorted({scene[4:6] for scene in test_scene_set}),
    }
    (output_dir / "scene_split.json").write_text(json.dumps(scene_split, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_source_scenes": len(scene_pairs),
        "num_train_scenes": len(train_scene_set),
        "num_test_scenes": len(test_scene_set),
        "train_months": sorted({scene[4:6] for scene in train_scene_set}),
        "test_months": sorted({scene[4:6] for scene in test_scene_set}),
        "patch_size": args.patch_size,
        "purity_threshold": args.purity_threshold,
        "train_stride_map": {str(k): int(v) for k, v in TRAIN_STRIDE_MAP.items()},
        "test_stride": TEST_STRIDE,
        "x_dtype": args.dtype,
        "y_dtype": "uint8",
        "label_map": {str(k): v for k, v in LABEL_MAP.items()},
        "excluded_labels": {str(k): v for k, v in EXCLUDED_LABELS.items()},
        "train_x_shape": list(train_x_shape),
        "train_y_shape": list(train_y_shape),
        "test_x_shape": list(test_x_shape),
        "test_y_shape": list(test_y_shape),
        "train_label_counts": {str(k): int(v) for k, v in sorted(train_counts.items())},
        "test_label_counts": {str(k): int(v) for k, v in sorted(test_counts.items())},
        "train_channel_mean": [float(x) for x in mean],
        "train_channel_std": [float(x) for x in std],
        "notes": [
            "Split by scene first, then cut patches.",
            "Both train and test contain scenes from months 01/04/07/10.",
            "Train uses dynamic overlapping stride; test uses fixed stride=64.",
            "Only CLP labels 0-4 are retained.",
            "Any patch containing FillValue in any FDI channel is discarded.",
            "Train normalization stats are applied to both train and test arrays.",
        ],
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output_dir, summary)

    print("Done.")
    print(f"Output folder: {output_dir}")
    print(f"Train samples: {sum(train_counts.values())}")
    print(f"Test samples: {sum(test_counts.values())}")


if __name__ == "__main__":
    main()
