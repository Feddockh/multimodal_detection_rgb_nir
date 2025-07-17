import argparse
import os
from pathlib import Path
import sys
from typing import Dict, List

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.v2 import (RandomHorizontalFlip,
                                        ColorJitter,
                                        RandomResize,
                                        RandomIoUCrop,
                                        Compose)
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# ---------------------------------------------------------------------------
# Project‑local imports
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.camera import Camera
from multicam_dataset import MultiCamDataset, SetType


# ---------------------------------------------------------------------------
# Dataloader helpers
# ---------------------------------------------------------------------------

def build_transforms():
    return Compose([
        RandomIoUCrop(min_scale=0.3, min_aspect_ratio=0.5, max_aspect_ratio=2.0),
        # RandomResize(min_size=480, max_size=1024),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])

def collate_fn(batch):
    """Merge a list of dataset outputs into model inputs."""
    images: List[torch.Tensor] = []
    targets: List[Dict[str, torch.Tensor]] = []

    for sample in batch:
        img, target = sample["firefly_left"]  # RGB camera only
        # BoundingBoxes → raw tensor if needed
        boxes = target["boxes"].tensor if hasattr(target["boxes"], "tensor") else target["boxes"]
        labels = target["labels"] + 1          #  ←←  SHIFT HERE
        images.append(img)
        targets.append({
            "boxes": boxes,
            "labels": labels,
        })
    return images, targets


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(num_classes: int) -> FasterRCNN:
    """Return a COCO‑pretrained Faster R‑CNN with a fresh class predictor."""
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")  # backbone encodes the RGB image
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, loader, device, epoch, accum_steps, lr_warmup_end, freeze_backbone_end):
    model.train()

    # Warm‑up LR linear ramp for very first epoch
    if epoch <= lr_warmup_end:
        lr_scale = epoch / max(1, lr_warmup_end)
        for g in optimizer.param_groups:
            g["lr"] = g.get("initial_lr", g["lr"]) * lr_scale

    # Freeze backbone early if requested
    # if epoch <= freeze_backbone_end:
    #     for p in model.backbone.parameters():
    #         p.requires_grad = False
    # elif epoch == freeze_backbone_end + 1:
    #     for p in model.backbone.parameters():
    #         p.requires_grad = True

    # Unfreeze backbone stages gradually
    if epoch <= freeze_backbone_end:
    # still fully frozen
        for p in model.backbone.parameters():
            p.requires_grad = False
    else:
        # unfreeze one additional stage each epoch
        stages = [model.backbone.body.layer4,
                model.backbone.body.layer3,
                model.backbone.body.layer2,
                model.backbone.body.layer1]          # order: deepest → shallowest
        n_open = min(epoch - freeze_backbone_end, len(stages))  # how many to unfreeze
        for i, stage in enumerate(stages):
            trainable = i < n_open
            for p in stage.parameters():
                p.requires_grad = trainable

    loss_total = 0.0
    optimizer.zero_grad()

    for step, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values()) / accum_steps
        loss_total += losses.item()

        losses.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

    print(f"Epoch {epoch}: train loss = {loss_total / len(loader):.4f}")


def evaluate_map(model, loader, device):
    """Compute COCO mAP using TorchMetrics and return the `map` scalar."""
    metric = MeanAveragePrecision()
    model.eval()

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            preds = model(images)  # list of dicts w/ boxes, scores, labels

            preds_cpu = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
            targets_cpu = [{k: v.detach().cpu() for k, v in t.items()} for t in targets]
            metric.update(preds_cpu, targets_cpu)

    stats = metric.compute()
    return stats  # dict with "map", "map_50", "map_75", etc.



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RGB‑only object detection trainer")
    parser.add_argument("--data_dir", default="data", type=str, help="Dataset root dir")
    parser.add_argument("--model_dir", default="models", type=str, help="Model save dir")
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--accum_steps", default=8, type=int,
                        help="Gradient accumulation steps to emulate larger batch")
    parser.add_argument("--lr_warmup_end", default=1, type=int,
                        help="Epochs of linear LR warm‑up")
    parser.add_argument("--freeze_backbone_end", default=4, type=int,
                        help="Last epoch with backbone frozen (inclusive)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Dataset ------------------------------------------------------------
    rgb_cam = Camera(name="firefly_left")
    cams = [rgb_cam]
    train_ds = MultiCamDataset(str(data_dir), cams, set_type=SetType.TRAIN, transforms=build_transforms())
    val_ds   = MultiCamDataset(str(data_dir), cams, set_type=SetType.VAL)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=4)

    # --- Model --------------------------------------------------------------
    num_classes = len(train_ds.get_class_names()) + 1  # +1 for background
    model = build_model(num_classes).to(args.device)

    # optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    for g in optimizer.param_groups:
        g["initial_lr"] = g["lr"]  # store base lr for warm‑up scaling

    # Cosine scheduler steps every epoch -------------------------------------
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training loop ------------------------------------------------------
    best_map = 0.0
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, optimizer, train_loader, args.device, epoch,
                        args.accum_steps, args.lr_warmup_end, args.freeze_backbone_end)

        lr_scheduler.step()

        stats = evaluate_map(model, val_loader, args.device)
        current_map = float(stats["map"])
        print(f"Epoch {epoch}: mAP@[.5:.95] = {current_map:.4f}, mAP@50 = {float(stats['map_50']):.4f}")

        # Save best
        if current_map > best_map:
            best_map = current_map
            torch.save(model.state_dict(), model_dir / "best_model_rgb.pt")
            print("Saved new best model")

    print("Done. Best mAP@[.5:.95]:", best_map)


if __name__ == "__main__":
    main()
