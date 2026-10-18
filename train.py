"""
SignVision Training Script
===========================
Trains a sign language recognition model using the configuration-driven
architecture system. Supports:
    - Dynamic class discovery from MP_Data folders (3 → 2000+ classes)
    - Multiple model architectures via configs/model_config.yaml
    - Mixed precision training (AMP) for RTX 3050 / low-VRAM GPUs
    - Cosine annealing LR schedule with early stopping
    - Checkpoint saving with full metadata for seamless inference

Usage:
    python train.py
    python train.py --data_dir dataset/MP_Data --epochs 100 --batch_size 16
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from src.data.dataset import SignLanguageDataset, create_dataloaders, pad_collate_fn
from src.data.augmentation import SequenceAugmentor
from src.models import build_model_from_config


def load_configs():
    """Load model and training YAML configs."""
    with open("configs/model_config.yaml", "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    with open("configs/training_config.yaml", "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    return model_cfg, train_cfg


def train(
    data_dir: str = None,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
):
    """Main training entry point."""

    model_cfg, train_cfg = load_configs()

    # CLI overrides take priority over YAML config
    data_dir = data_dir or train_cfg["data"].get("data_path", "MP_Data")
    epochs = epochs or train_cfg["training"].get("epochs", 200)
    batch_size = batch_size or train_cfg["training"].get("batch_size", 32)
    learning_rate = learning_rate or train_cfg["training"].get("learning_rate", 0.001)
    max_frames = train_cfg["data"].get("max_sequence_length", 90)
    num_workers = train_cfg["data"].get("num_workers", 4)
    pin_memory = train_cfg["data"].get("pin_memory", True)
    use_amp = train_cfg["training"].get("mixed_precision", True)
    patience = train_cfg["training"].get("early_stopping_patience", 20)
    grad_clip = train_cfg["training"].get("gradient_clip_norm", 1.0)
    label_smoothing = train_cfg["training"].get("label_smoothing", 0.1)
    seed = train_cfg.get("seed", 42)
    checkpoint_dir = train_cfg["logging"].get("checkpoint_dir", "checkpoints")

    torch.manual_seed(seed)

    # ---- Device ----
    device_str = train_cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[Train] Device: {device}")

    # ---- Augmentation ----
    aug_cfg = train_cfg.get("augmentation", {})
    augmentor = SequenceAugmentor(aug_cfg) if aug_cfg.get("enabled", True) else None

    # ---- Data ----
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        data_dir=data_dir,
        max_frames=max_frames,
        batch_size=batch_size,
        train_split=train_cfg["data"].get("train_split", 0.8),
        val_split=train_cfg["data"].get("val_split", 0.1),
        test_split=train_cfg["data"].get("test_split", 0.1),
        num_workers=num_workers,
        pin_memory=pin_memory,
        augmentor=augmentor,
        seed=seed,
    )

    num_classes = dataset.num_classes
    actions = dataset.classes
    active_model = model_cfg.get("active_model", "bilstm_attention")

    os.makedirs(checkpoint_dir, exist_ok=True)
    dataset.save_label_map(os.path.join(checkpoint_dir, "label_map.json"))
    with open("action_labels.json", "w", encoding="utf-8") as f:
        json.dump(list(actions), f, indent=2)

    print(f"[Train] Classes: {num_classes}")
    print(f"[Train] Samples: {len(dataset)}")
    print(f"[Train] Architecture: {active_model}")
    print(f"[Train] Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")


    # ---- Model ----
    model = build_model_from_config(model_cfg, num_classes=num_classes)
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Trainable parameters: {param_count:,}")

    # ---- Optimizer ----
    opt_name = train_cfg["training"].get("optimizer", "adamw")
    weight_decay = train_cfg["training"].get("weight_decay", 0.0001)
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    # ---- Scheduler ----
    sched_name = train_cfg["training"].get("scheduler", "cosine_annealing")
    sched_params = train_cfg["training"].get("scheduler_params", {})
    if sched_name == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_params.get("T_max", epochs),
            eta_min=sched_params.get("eta_min", 1e-5),
        )
    elif sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # ---- Loss ----
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # ---- AMP ----
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None

    # ---- Training loop ----
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # -- Train --
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for seqs, labels, mask in loop:
            seqs = seqs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(seqs, mask=mask)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(seqs, mask=mask)
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            train_loss += loss.item() * seqs.size(0)
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            loop.set_postfix(loss=loss.item(), acc=f"{100.*train_correct/train_total:.1f}%")

        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total

        # -- Validate --
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for seqs, labels, mask in val_loader:
                seqs = seqs.to(device)
                labels = labels.to(device)
                mask = mask.to(device)

                logits = model(seqs, mask=mask)
                loss = criterion(logits, labels)

                val_loss += loss.item() * seqs.size(0)
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= max(val_total, 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)

        # -- Scheduler step --
        if sched_name == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | "
            f"LR: {current_lr:.6f}"
        )

        # -- Checkpoint --
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "actions": actions,
                "num_classes": num_classes,
                "active_model": active_model,
                "classes": actions,
            }
            ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(ckpt, ckpt_path)
            print(f"  --> Saved best checkpoint (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered after {patience} epochs without improvement.")
                break

    # -- Save final model --
    final_ckpt = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch + 1,
        "actions": actions,
        "num_classes": num_classes,
        "active_model": active_model,
        "classes": actions,
    }
    torch.save(final_ckpt, os.path.join(checkpoint_dir, "last_model.pt"))
    print(f"\n[Train] Complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SignVision model")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
