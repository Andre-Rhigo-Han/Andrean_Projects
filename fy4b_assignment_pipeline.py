from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

CLASS_NAMES = ["Clear", "Water", "SuperCooled", "Mixed", "Ice"]


@dataclass
class AssignmentConfig:
    data_dir: Path = Path("data_processed/fy4b_vit_month_stratified_scene_split")
    runs_dir: Path = Path("runs/assignment_suite")
    attention_dir: Path = Path("attention_maps")
    tradeoff_png: Path = Path("speed_accuracy_tradeoff.png")
    results_csv: Path = Path("assignment_results.csv")
    results_json: Path = Path("assignment_results.json")

    img_size: int = 64
    in_channels: int = 15
    num_classes: int = 5
    class_names: list[str] = field(default_factory=lambda: CLASS_NAMES.copy())

    seed: int = 42
    val_scene_ratio: float = 0.2
    val_split_trials: int = 2000

    vit_patch_size: int = 8
    vit_embed_dim: int = 256
    vit_depth: int = 6
    vit_num_heads: int = 8
    vit_mlp_ratio: float = 3.0
    vit_dropout: float = 0.1

    batch_size: int = 128
    num_epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 5e-2
    label_smoothing: float = 0.05
    early_stopping_patience: int = 15

    num_workers: int = 0
    pin_memory: bool = torch.cuda.is_available()
    use_amp: bool = torch.cuda.is_available()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_jsonable_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ["data_dir", "runs_dir", "attention_dir", "tradeoff_png", "results_csv", "results_json"]:
            out[key] = str(out[key])
        return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@dataclass
class DataBundle:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    train_meta: pd.DataFrame
    test_meta: pd.DataFrame
    train_scene_manifest: pd.DataFrame


@dataclass
class SplitBundle:
    subtrain_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    subtrain_scenes: set[str]
    val_scenes: set[str]
    class_weights: np.ndarray


def load_data_bundle(data_dir: Path) -> DataBundle:
    required = {
        "train_x": data_dir / "train_x_fy4b.npy",
        "train_y": data_dir / "train_y_fy4b.npy",
        "test_x": data_dir / "test_x_fy4b.npy",
        "test_y": data_dir / "test_y_fy4b.npy",
        "train_meta": data_dir / "train_patch_metadata.csv",
        "test_meta": data_dir / "test_patch_metadata.csv",
        "train_scene_manifest": data_dir / "train_scene_manifest.csv",
    }
    for path in required.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    return DataBundle(
        train_x=np.load(required["train_x"], mmap_mode="r"),
        train_y=np.load(required["train_y"], mmap_mode="r"),
        test_x=np.load(required["test_x"], mmap_mode="r"),
        test_y=np.load(required["test_y"], mmap_mode="r"),
        train_meta=pd.read_csv(required["train_meta"]),
        test_meta=pd.read_csv(required["test_meta"]),
        train_scene_manifest=pd.read_csv(required["train_scene_manifest"]),
    )


def select_val_scenes(train_meta: pd.DataFrame, num_classes: int, val_ratio: float, seed: int, trials: int) -> tuple[set[str], np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    scene_class_counts = defaultdict(lambda: np.zeros(num_classes, dtype=np.int64))
    scene_month: dict[str, str] = {}
    for _, row in train_meta.iterrows():
        scene = str(row["start_time"])
        label = int(row["label"])
        month = str(row["month"]).zfill(2)
        scene_class_counts[scene][label] += 1
        scene_month[scene] = month

    scenes = sorted(scene_class_counts.keys())
    by_month = defaultdict(list)
    for scene in scenes:
        by_month[scene_month[scene]].append(scene)

    total_counts = sum((scene_class_counts[s] for s in scenes), np.zeros(num_classes, dtype=np.int64))
    month_pick: dict[str, int] = {}
    for month, scene_list in by_month.items():
        k = max(1, round(len(scene_list) * val_ratio))
        k = min(k, len(scene_list) - 1) if len(scene_list) > 1 else 1
        month_pick[month] = k

    best: tuple[float, list[str], np.ndarray, np.ndarray] | None = None
    scene_pool_by_month = {m: list(v) for m, v in by_month.items()}
    for _ in range(trials):
        candidate: set[str] = set()
        for month, scene_list in scene_pool_by_month.items():
            candidate.update(rng.sample(scene_list, month_pick[month]))

        val_counts = sum((scene_class_counts[s] for s in candidate), np.zeros(num_classes, dtype=np.int64))
        train_counts = total_counts - val_counts
        penalty = 0.0
        if np.any(val_counts == 0):
            penalty += 1e6
        if np.any(train_counts == 0):
            penalty += 1e6
        target = total_counts * val_ratio
        penalty += float(np.mean(((val_counts - target) / np.maximum(target, 1)) ** 2)) * 1000
        penalty += float(np.mean(((train_counts - (total_counts - target)) / np.maximum(total_counts - target, 1)) ** 2)) * 1000
        if best is None or penalty < best[0]:
            best = (penalty, sorted(candidate), val_counts.copy(), train_counts.copy())

    assert best is not None
    return set(best[1]), best[2], best[3]


def build_splits(bundle: DataBundle, cfg: AssignmentConfig) -> SplitBundle:
    val_scenes, _, _ = select_val_scenes(bundle.train_meta, cfg.num_classes, cfg.val_scene_ratio, cfg.seed, cfg.val_split_trials)
    train_scenes = set(bundle.train_meta["start_time"].astype(str).unique())
    subtrain_scenes = train_scenes - val_scenes
    subtrain_indices = bundle.train_meta.loc[bundle.train_meta["start_time"].astype(str).isin(subtrain_scenes), "sample_index"].to_numpy(dtype=np.int64)
    val_indices = bundle.train_meta.loc[bundle.train_meta["start_time"].astype(str).isin(val_scenes), "sample_index"].to_numpy(dtype=np.int64)
    test_indices = bundle.test_meta["sample_index"].to_numpy(dtype=np.int64)
    subtrain_labels = bundle.train_y[subtrain_indices].astype(np.int64, copy=False)
    train_label_counts = np.bincount(subtrain_labels, minlength=cfg.num_classes)
    class_weights = train_label_counts.sum() / np.maximum(train_label_counts, 1)
    class_weights = class_weights / class_weights.mean()
    return SplitBundle(subtrain_indices, val_indices, test_indices, subtrain_scenes, val_scenes, class_weights.astype(np.float32))


def stratified_subsample_indices(indices: np.ndarray, labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if fraction >= 0.999999:
        return np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    for cls in sorted(np.unique(labels).tolist()):
        cls_indices = np.asarray(indices[labels == cls], dtype=np.int64)
        k = max(1, int(round(len(cls_indices) * fraction)))
        k = min(k, len(cls_indices))
        chosen = rng.choice(cls_indices, size=k, replace=False)
        kept.append(np.sort(chosen))
    merged = np.concatenate(kept)
    rng.shuffle(merged)
    return merged.astype(np.int64)


class SatelliteTrainAugment:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[1])
        k = int(torch.randint(0, 4, (1,)).item())
        x = torch.rot90(x, k, dims=[1, 2])
        return x.contiguous()


class FY4BNpyDataset(Dataset):
    def __init__(self, x_array: np.ndarray, y_array: np.ndarray, indices: np.ndarray, transform=None):
        self.x_array = x_array
        self.y_array = y_array
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        real_idx = int(self.indices[idx])
        x = np.array(self.x_array[real_idx], dtype=np.float32, copy=True)
        y = int(self.y_array[real_idx])
        x_t = torch.from_numpy(x)
        if self.transform is not None:
            x_t = self.transform(x_t)
        return x_t, y


def make_loaders(bundle: DataBundle, split_bundle: SplitBundle, cfg: AssignmentConfig, fraction: float) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    subtrain_labels = bundle.train_y[split_bundle.subtrain_indices].astype(np.int64, copy=False)
    fraction_indices = stratified_subsample_indices(split_bundle.subtrain_indices, subtrain_labels, fraction, cfg.seed)
    train_dataset = FY4BNpyDataset(bundle.train_x, bundle.train_y, fraction_indices, transform=SatelliteTrainAugment())
    val_dataset = FY4BNpyDataset(bundle.train_x, bundle.train_y, split_bundle.val_indices, transform=None)
    test_dataset = FY4BNpyDataset(bundle.test_x, bundle.test_y, split_bundle.test_indices, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    return train_loader, val_loader, test_loader, fraction_indices

class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int = 64, patch_size: int = 8, in_channels: int = 15, embed_dim: int = 256):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        bsz, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(bsz, tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(bsz, tokens, channels)
        out = self.proj_drop(self.proj(out))
        return (out, attn) if return_attention else out


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 3.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        if return_attention:
            attn_out, attn = self.attn(self.norm1(x), return_attention=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FY4BViT(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=15, num_classes=5, embed_dim=256, depth=6, num_heads=8, mlp_ratio=3.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(bsz, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def get_last_selfattention(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(bsz, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks[:-1]:
            x = block(x)
        _, attn = self.blocks[-1](x, return_attention=True)
        return attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.head(x[:, 0])


def build_vit(cfg: AssignmentConfig) -> nn.Module:
    return FY4BViT(cfg.img_size, cfg.vit_patch_size, cfg.in_channels, cfg.num_classes, cfg.vit_embed_dim, cfg.vit_depth, cfg.vit_num_heads, cfg.vit_mlp_ratio, cfg.vit_dropout)


def build_cnn(cfg: AssignmentConfig) -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(cfg.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
    return model


def build_model(model_name: str, cfg: AssignmentConfig) -> nn.Module:
    if model_name == "vit":
        return build_vit(cfg)
    if model_name == "cnn":
        return build_cnn(cfg)
    raise ValueError(f"Unsupported model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def evaluate_predictions(targets: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    return {
        "acc": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro")),
        "weighted_f1": float(f1_score(targets, preds, average="weighted")),
    }


def run_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, *, optimizer=None, scaler=None, use_amp=False) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    losses: list[float] = []
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    start = time.perf_counter()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        amp_ctx = autocast(enabled=use_amp) if device.type == "cuda" else nullcontext()
        with torch.set_grad_enabled(is_train):
            with amp_ctx:
                logits = model(images)
                loss = criterion(logits, labels)
            if is_train:
                if scaler is not None and use_amp and device.type == "cuda":
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
        losses.append(float(loss.item()))
        preds = logits.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(labels.detach().cpu().numpy())
    elapsed = time.perf_counter() - start
    preds_np = np.concatenate(all_preds)
    targets_np = np.concatenate(all_targets)
    scores = evaluate_predictions(targets_np, preds_np)
    return {
        "loss": float(np.mean(losses)), "acc": scores["acc"], "macro_f1": scores["macro_f1"], "weighted_f1": scores["weighted_f1"],
        "preds": preds_np, "targets": targets_np, "time_sec": float(elapsed),
    }


def save_history_plot(history: dict[str, list[float]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10)); axes = axes.ravel()
    axes[0].plot(history["train_loss"], label="train"); axes[0].plot(history["val_loss"], label="val"); axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history["train_acc"], label="train"); axes[1].plot(history["val_acc"], label="val"); axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True)
    axes[2].plot(history["train_macro_f1"], label="train"); axes[2].plot(history["val_macro_f1"], label="val"); axes[2].set_title("Macro-F1"); axes[2].legend(); axes[2].grid(True)
    axes[3].plot(history["epoch_time_sec"], label="epoch"); axes[3].plot(history["train_epoch_time_sec"], label="train"); axes[3].plot(history["val_epoch_time_sec"], label="val"); axes[3].set_title("Timing (seconds)"); axes[3].legend(); axes[3].grid(True)
    fig.tight_layout(); fig.savefig(output_path, dpi=180); plt.close(fig)


def save_confusion_plot(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], title: str, output_path: Path) -> None:
    fig = plt.figure(figsize=(6, 5)); sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title); plt.tight_layout(); fig.savefig(output_path, dpi=180); plt.close(fig)


def load_checkpoint_state(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint
    if isinstance(checkpoint, dict):
        return {"model_state_dict": checkpoint}
    raise ValueError(f"Unsupported checkpoint format: {path}")

def train_single_experiment(model_name: str, fraction: float, bundle: DataBundle, split_bundle: SplitBundle, cfg: AssignmentConfig, *, skip_existing: bool = False) -> tuple[dict[str, Any], Path]:
    fraction_tag = f"{int(round(fraction * 100)):03d}"
    run_dir = cfg.runs_dir / f"{model_name}_fraction_{fraction_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "best_model.pth"
    if skip_existing and summary_path.exists() and checkpoint_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8")), checkpoint_path

    device = torch.device(cfg.device)
    set_seed(cfg.seed)
    model = build_model(model_name, cfg).to(device)
    param_count = count_parameters(model)
    train_loader, val_loader, test_loader, fraction_indices = make_loaders(bundle, split_bundle, cfg, fraction)
    train_fraction_labels = bundle.train_y[fraction_indices].astype(np.int64, copy=False)
    train_fraction_counts = np.bincount(train_fraction_labels, minlength=cfg.num_classes)
    fraction_class_weights = train_fraction_counts.sum() / np.maximum(train_fraction_counts, 1)
    fraction_class_weights = fraction_class_weights / fraction_class_weights.mean()
    class_weights_t = torch.tensor(fraction_class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=cfg.label_smoothing)
    criterion_eval = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    scaler = GradScaler(enabled=cfg.use_amp and device.type == "cuda")

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_macro_f1": [], "val_macro_f1": [],
        "train_weighted_f1": [], "val_weighted_f1": [], "train_epoch_time_sec": [], "val_epoch_time_sec": [], "epoch_time_sec": [], "lr": [],
    }
    best_val_f1 = -1.0; best_epoch = -1; patience_counter = 0; total_train_start = time.perf_counter()

    for epoch in range(cfg.num_epochs):
        epoch_start = time.perf_counter()
        train_metrics = run_one_epoch(model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler, use_amp=cfg.use_amp)
        val_metrics = run_one_epoch(model, val_loader, criterion_eval, device, use_amp=False)
        scheduler.step(); epoch_elapsed = time.perf_counter() - epoch_start

        history["train_loss"].append(train_metrics["loss"]); history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"]); history["val_acc"].append(val_metrics["acc"])
        history["train_macro_f1"].append(train_metrics["macro_f1"]); history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["train_weighted_f1"].append(train_metrics["weighted_f1"]); history["val_weighted_f1"].append(val_metrics["weighted_f1"])
        history["train_epoch_time_sec"].append(train_metrics["time_sec"]); history["val_epoch_time_sec"].append(val_metrics["time_sec"])
        history["epoch_time_sec"].append(float(epoch_elapsed)); history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]; best_epoch = epoch + 1; patience_counter = 0
            torch.save({"epoch": epoch + 1, "model_name": model_name, "fraction": fraction, "model_state_dict": model.state_dict(), "val_macro_f1": best_val_f1, "config": cfg.to_jsonable_dict()}, checkpoint_path)
        else:
            patience_counter += 1

        print(f"[{model_name} {fraction:.0%}] Epoch {epoch + 1:03d}/{cfg.num_epochs} | train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} | train_acc={train_metrics['acc']:.4f} val_acc={val_metrics['acc']:.4f} | train_f1={train_metrics['macro_f1']:.4f} val_f1={val_metrics['macro_f1']:.4f} | epoch_time={epoch_elapsed:.2f}s")
        if patience_counter >= cfg.early_stopping_patience:
            print(f"[{model_name} {fraction:.0%}] Early stopping at epoch {epoch + 1}")
            break

    total_training_time_sec = float(time.perf_counter() - total_train_start)
    checkpoint = load_checkpoint_state(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    val_metrics = run_one_epoch(model, val_loader, criterion_eval, device, use_amp=False)
    test_metrics = run_one_epoch(model, test_loader, criterion_eval, device, use_amp=False)

    save_history_plot(history, run_dir / "history.png")
    save_confusion_plot(val_metrics["targets"], val_metrics["preds"], cfg.class_names, f"{model_name.upper()} Val Confusion Matrix ({fraction:.0%})", run_dir / "val_confusion_matrix.png")
    save_confusion_plot(test_metrics["targets"], test_metrics["preds"], cfg.class_names, f"{model_name.upper()} Test Confusion Matrix ({fraction:.0%})", run_dir / "test_confusion_matrix.png")

    history_json = history | {"total_training_time_sec": total_training_time_sec}
    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history_json, f, ensure_ascii=False, indent=2)

    summary = {
        "model_name": model_name, "fraction": float(fraction), "fraction_percent": int(round(fraction * 100)), "run_dir": str(run_dir), "checkpoint_path": str(checkpoint_path),
        "parameter_count": int(param_count), "train_samples": int(len(fraction_indices)), "val_samples": int(len(split_bundle.val_indices)), "test_samples": int(len(split_bundle.test_indices)),
        "train_label_counts": {str(i): int(v) for i, v in enumerate(train_fraction_counts.tolist())}, "best_epoch": int(checkpoint.get("epoch", best_epoch)), "best_val_macro_f1": float(checkpoint.get("val_macro_f1", best_val_f1)),
        "avg_epoch_time_sec": float(np.mean(history["epoch_time_sec"])), "total_training_time_sec": total_training_time_sec,
        "val_loss": float(val_metrics["loss"]), "val_acc": float(val_metrics["acc"]), "val_macro_f1": float(val_metrics["macro_f1"]), "val_weighted_f1": float(val_metrics["weighted_f1"]),
        "test_loss": float(test_metrics["loss"]), "test_acc": float(test_metrics["acc"]), "test_macro_f1": float(test_metrics["macro_f1"]), "test_weighted_f1": float(test_metrics["weighted_f1"]),
        "val_classification_report": classification_report(val_metrics["targets"], val_metrics["preds"], target_names=cfg.class_names, digits=4, output_dict=True),
        "test_classification_report": classification_report(test_metrics["targets"], test_metrics["preds"], target_names=cfg.class_names, digits=4, output_dict=True),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary, checkpoint_path


def save_results_table(results: list[dict[str, Any]], cfg: AssignmentConfig) -> pd.DataFrame:
    if not results:
        raise ValueError("No results to save.")
    df = pd.DataFrame(results).sort_values(["model_name", "fraction"], ascending=[True, False]).reset_index(drop=True)
    df.to_csv(cfg.results_csv, index=False, encoding="utf-8-sig")
    cfg.results_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return df


def plot_speed_accuracy_tradeoff(results: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    markers = {"vit": "o", "cnn": "s"}; colors = {"vit": "#1f77b4", "cnn": "#d62728"}
    for row in results:
        model_name = str(row["model_name"]); x = float(row["total_training_time_sec"]); y = float(row["test_acc"])
        label = f"{model_name.upper()}-{int(row['fraction_percent'])}%\nF1={row['test_macro_f1']:.3f}"
        ax.scatter(x, y, s=140, marker=markers.get(model_name, "o"), color=colors.get(model_name, "#333333"), alpha=0.9)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 5), fontsize=9)
    ax.set_xlabel("Total training time (seconds)"); ax.set_ylabel("Test accuracy"); ax.set_title("Speed-Accuracy Tradeoff (FY4B ViT vs CNN)"); ax.grid(True, alpha=0.3)
    handles = [plt.Line2D([0], [0], marker=markers.get(m, "o"), color="w", markerfacecolor=colors.get(m, "#333333"), markersize=10, label=m.upper()) for m in sorted({str(r['model_name']) for r in results})]
    ax.legend(handles=handles); fig.tight_layout(); fig.savefig(output_path, dpi=220); plt.close(fig)

def normalize_to_01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    img_min = float(img.min()); img_max = float(img.max())
    if img_max - img_min < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return (img - img_min) / (img_max - img_min)


def build_display_image(sample: np.ndarray) -> np.ndarray:
    vis_channels = sample[:3] if sample.shape[0] >= 3 else sample[:1]
    if vis_channels.shape[0] == 1:
        gray = normalize_to_01(vis_channels[0]); return np.stack([gray, gray, gray], axis=-1)
    rgb = np.stack([normalize_to_01(vis_channels[i]) for i in range(3)], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def choose_attention_samples(model: FY4BViT, bundle: DataBundle, split_bundle: SplitBundle, cfg: AssignmentConfig, *, max_per_class: int = 1) -> list[dict[str, Any]]:
    device = torch.device(cfg.device)
    selected: list[dict[str, Any]] = []; counts = Counter()
    for sample_index in split_bundle.test_indices.tolist():
        label = int(bundle.test_y[sample_index])
        if counts[label] >= max_per_class:
            continue
        sample_np = np.array(bundle.test_x[sample_index], dtype=np.float32, copy=True)
        sample_tensor = torch.from_numpy(sample_np).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(model(sample_tensor).argmax(dim=1).item())
        if pred != label:
            continue
        meta_row = bundle.test_meta.loc[bundle.test_meta["sample_index"] == sample_index].iloc[0].to_dict()
        selected.append({"sample_index": sample_index, "label": label, "pred": pred, "x": sample_np, "meta": meta_row})
        counts[label] += 1
        if all(counts[c] >= max_per_class for c in range(cfg.num_classes)):
            break
    return selected


def generate_attention_maps(checkpoint_path: Path, bundle: DataBundle, split_bundle: SplitBundle, cfg: AssignmentConfig, *, output_dir: Path | None = None) -> list[dict[str, Any]]:
    device = torch.device(cfg.device); out_dir = output_dir or cfg.attention_dir; out_dir.mkdir(parents=True, exist_ok=True)
    model = build_vit(cfg).to(device); checkpoint = load_checkpoint_state(checkpoint_path); model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    samples = choose_attention_samples(model, bundle, split_bundle, cfg, max_per_class=1); records: list[dict[str, Any]] = []
    for item in samples:
        sample_index = int(item["sample_index"]); label = int(item["label"]); x = item["x"]; meta = item["meta"]
        tensor = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor); probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy(); attn = model.get_last_selfattention(tensor)
        patch_grid = cfg.img_size // cfg.vit_patch_size
        cls_attn = attn[0, :, 0, 1:].mean(dim=0).reshape(1, 1, patch_grid, patch_grid)
        attn_np = F.interpolate(cls_attn, size=(cfg.img_size, cfg.img_size), mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
        attn_np = normalize_to_01(attn_np); display_img = build_display_image(x)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(display_img); axes[0].set_title("Input patch (vis ch 1-3)"); axes[0].axis("off")
        axes[1].imshow(attn_np, cmap="inferno"); axes[1].set_title("Last-layer CLS attention"); axes[1].axis("off")
        axes[2].imshow(display_img); axes[2].imshow(attn_np, cmap="jet", alpha=0.45); axes[2].set_title(f"Overlay\ntrue={cfg.class_names[label]} pred={cfg.class_names[int(np.argmax(probs))]}"); axes[2].axis("off")
        file_name = f"class_{label}_{cfg.class_names[label].lower()}_sample_{sample_index}_scene_{meta['start_time']}_r{int(meta['row_start'])}_c{int(meta['col_start'])}.png"
        fig.tight_layout(); fig.savefig(out_dir / file_name, dpi=220); plt.close(fig)
        records.append({"file": file_name, "sample_index": sample_index, "scene": str(meta["start_time"]), "row_start": int(meta["row_start"]), "col_start": int(meta["col_start"]), "true_label": cfg.class_names[label], "pred_label": cfg.class_names[int(np.argmax(probs))], "pred_confidence": float(np.max(probs))})
    with (out_dir / "attention_map_index.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return records


def run_full_suite(args: argparse.Namespace) -> None:
    cfg = AssignmentConfig()
    cfg.num_epochs = args.epochs if args.epochs is not None else cfg.num_epochs
    cfg.batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size
    cfg.data_dir = Path(args.data_dir) if args.data_dir else cfg.data_dir
    cfg.runs_dir = Path(args.runs_dir) if args.runs_dir else cfg.runs_dir
    cfg.attention_dir = Path(args.attention_dir) if args.attention_dir else cfg.attention_dir
    cfg.tradeoff_png = Path(args.tradeoff_png) if args.tradeoff_png else cfg.tradeoff_png
    cfg.results_csv = Path(args.results_csv) if args.results_csv else cfg.results_csv
    cfg.results_json = Path(args.results_json) if args.results_json else cfg.results_json
    bundle = load_data_bundle(cfg.data_dir); split_bundle = build_splits(bundle, cfg)
    results: list[dict[str, Any]] = []; checkpoint_map: dict[tuple[str, int], Path] = {}
    for model_name in args.models:
        for fraction in args.fractions:
            summary, checkpoint_path = train_single_experiment(model_name, float(fraction), bundle, split_bundle, cfg, skip_existing=args.skip_existing)
            results.append(summary); checkpoint_map[(model_name, int(round(float(fraction) * 100)))] = checkpoint_path
    save_results_table(results, cfg); plot_speed_accuracy_tradeoff(results, cfg.tradeoff_png)
    attention_ckpt: Path | None = Path(args.attention_checkpoint) if args.attention_checkpoint else checkpoint_map.get(("vit", 100))
    if attention_ckpt is None:
        existing = Path("runs/vit_fy4b_15ch/best_vit_fy4b.pth")
        attention_ckpt = existing if existing.exists() else None
    if attention_ckpt is not None and attention_ckpt.exists():
        generate_attention_maps(attention_ckpt, bundle, split_bundle, cfg, output_dir=cfg.attention_dir)
    else:
        print("No ViT checkpoint available for attention-map generation.")


def run_attention_only(args: argparse.Namespace) -> None:
    cfg = AssignmentConfig(); cfg.data_dir = Path(args.data_dir) if args.data_dir else cfg.data_dir; cfg.attention_dir = Path(args.output_dir) if args.output_dir else cfg.attention_dir
    bundle = load_data_bundle(cfg.data_dir); split_bundle = build_splits(bundle, cfg); records = generate_attention_maps(Path(args.checkpoint), bundle, split_bundle, cfg, output_dir=cfg.attention_dir)
    print(f"Saved {len(records)} attention-map figures to {cfg.attention_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FY4B assignment pipeline: ViT/CNN comparison, timing, ablations, and attention maps.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    full = subparsers.add_parser("full", help="Run the full assignment suite.")
    full.add_argument("--data-dir", default=None); full.add_argument("--runs-dir", default=None); full.add_argument("--attention-dir", default=None)
    full.add_argument("--tradeoff-png", default=None); full.add_argument("--results-csv", default=None); full.add_argument("--results-json", default=None)
    full.add_argument("--epochs", type=int, default=None); full.add_argument("--batch-size", type=int, default=None)
    full.add_argument("--models", nargs="+", default=["vit", "cnn"], choices=["vit", "cnn"])
    full.add_argument("--fractions", nargs="+", type=float, default=[1.0, 0.2, 0.1])
    full.add_argument("--skip-existing", action="store_true")
    full.add_argument("--attention-checkpoint", default=None)
    attention = subparsers.add_parser("attention", help="Generate attention maps from an existing ViT checkpoint.")
    attention.add_argument("--checkpoint", required=True); attention.add_argument("--data-dir", default=None); attention.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    parser = build_parser(); args = parser.parse_args()
    if args.command == "full":
        run_full_suite(args)
    elif args.command == "attention":
        run_attention_only(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
