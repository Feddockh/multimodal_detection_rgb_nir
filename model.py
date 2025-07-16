# models/yolo_rgb.py
import torch, torch.nn as nn
from ultralytics.nn.tasks import DetectionModel           # backbone+neck+head
from ultralytics.utils.loss import v8DetectionLoss            # built-in YOLOv8 loss
from ultralytics.utils.torch_utils import initialize_weights
from ultralytics.utils.checks import check_yaml
from pathlib import Path


class YOLOv8RGB(nn.Module):
    """
    Wrapper around Ultralytics DetectionModel that:
      • builds from .yaml,
      • lets you override nc,
      • loads COCO pretrained .pt (optional).
    """
    def __init__(self,
                 cfg_yaml="yolov8s.yaml",
                 pretrained_ckpt="yolov8s.pt",
                 num_classes=2,
                 device="cuda"):
        super().__init__()
        cfg_yaml = check_yaml(cfg_yaml)            # resolve path
        self.model = DetectionModel(cfg_yaml, ch=3, nc=num_classes, verbose=False)

        if pretrained_ckpt:                        # load weights except cls layer
            state = torch.load(pretrained_ckpt, map_location="cpu")["model"].float().state_dict()
            self.model.load_state_dict(state, strict=False)

        initialize_weights(self.model)             # inits any newly added layers
        self.loss_fn = v8DetectionLoss(self.model) # anchor-free v8 loss
        self.device  = torch.device(device)
        self.to(self.device)

    def forward(self, x):
        # returns list(P3,P4,P5) predictions  [B, anchors, 85] by default
        return self.model(x)

    def loss(self, preds, targets):
        # Ultralytics’ targets format: list[L_i] of tensors per image
        return self.loss_fn(preds, targets)[0]     # returns (total_loss, metrics)
