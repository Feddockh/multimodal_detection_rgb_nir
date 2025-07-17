import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics.data.loaders import LoadImagesAndLabels

class MultiModalLoader(LoadImagesAndLabels):
    """
    A YOLOv8 loader that knows about an RGB+NIR pair folder structure,
    but for now only returns the RGB image and labels.

    Assumes your YAML `data:` path points at the root containing:
      ├─ firefly_left/images/...
      └─ ximea/images/...
    """

    def __init__(self, path, *args, **kwargs):
        """
        Args:
            path (str): same as in your data YAML; e.g. "../data"
            *args, **kwargs: passed to LoadImagesAndLabels (imgsz, batch, augment, etc.)
        """
        super().__init__(path, *args, **kwargs)
        # Derive the NIR base by swapping directory names.
        # Adjust these substrings to match your on-disk layout.
        self.rgb_root = Path(self.path)
        self.nir_root = Path(str(self.rgb_root).replace("firefly_left", "ximea"))

    def __getitem__(self, index):
        # Load RGB + labels exactly as YOLOv8 would:
        im_rgb, labels, paths, shapes, pads = super().__getitem__(index)

        # Derive the matching NIR path (for future use):
        rgb_path = Path(paths[0] if isinstance(paths, (list, tuple)) else paths)
        nir_path = rgb_path.with_name(rgb_path.name).with_parent(
            str(rgb_path.parent).replace("firefly_left/images", "ximea/images")
        )
        # Example: load NIR (H×W×3 BGR float32 0–1)
        # nir = cv2.imread(str(nir_path))
        # nir = nir[:, :, ::-1]  # BGR→RGB
        # nir = torch.from_numpy(nir).permute(2, 0, 1).float() / 255.0

        # For now, we ignore nir and return only RGB so training is unchanged:
        return im_rgb, labels, paths, shapes, pads

























































# # ---------------------------------------------------------------------
# # rgb_nir_yolov8.py
# # ---------------------------------------------------------------------
# import copy, os, sys
# from typing import List, Dict

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader

# # Add the parent directory to the system path to import the dataset module
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from multicam_dataset import MultiCamDataset, SetType, Camera

# from ultralytics import YOLO                       # high-level wrapper
# from ultralytics.nn.modules import Detect          # Detect head class
# from ultralytics.utils.loss import v8DetectionLoss   # official v8 loss
# from ultralytics.data.loaders import LoadImagesAndLabels


# class InputMode:
#     RGB = 0
#     NIR = 1
#     FUSION = 2

# class YOLOv8RGBNIR(nn.Module):
#     def __init__(self,
#                  base_weights: str = "yolov8n.pt",
#                  mode: InputMode = InputMode.FUSION,
#                  nc: int = 2):
        
#         # Initialize the parent class
#         super().__init__()

#         # Save the mode and number of classes
#         self.mode = mode
#         self.nc = nc

#         # Load a pre-trained YOLOv8 model
#         base = YOLO(base_weights).model

#         # Copy the backbone and detection head from the base model
#         backbone_neck = nn.Sequential(*base.model[:-1])
#         detect_head   = copy.deepcopy(base.model[-1])

#         if mode == InputMode.FUSION:
#             self.rgb_backbone_neck = backbone_neck                 # 3-chan
#             self.nir_backbone_neck = copy.deepcopy(backbone_neck)  # 1-chan
            
#             # change first conv of NIR path to accept 1 channel
#             first_conv = next(m for m in self.nir_backbone_neck.modules()
#                               if isinstance(m, nn.Conv2d))
#             first_conv.in_channels = 1
#             first_conv.weight = nn.Parameter(first_conv.weight.sum(1, keepdim=True))

#             # after backbone+neck YOLOv8 produces 3 feature maps
#             ch_rgb = detect_head.stride  # just to grab #maps
#             ch_list = detect_head.channels  # e.g. [256, 512, 1024]
#             fused_ch = [c * 2 for c in ch_list]  # concat => double channels

#             # little 1×1 Conv to shrink concatenated feature maps back
#             shrink_layers = nn.ModuleList(
#                 nn.Conv2d(f, c, 1, bias=False) for f, c in zip(fused_ch, ch_list)
#             )
#             self.shrink = shrink_layers
#         else:  # single-modality
#             if mode == InputMode.RGB:
#                 self.backbone_neck = backbone_neck         # RGB weights OK
#             else:  # NIR
#                 self.backbone_neck = copy.deepcopy(backbone_neck)
#                 first_conv = next(m for m in self.backbone_neck.modules()
#                                   if isinstance(m, nn.Conv2d))
#                 first_conv.in_channels = 1
#                 first_conv.weight = nn.Parameter(first_conv.weight.sum(1, keepdim=True))

#         # ------------ Detect head – need fresh instance for nc -----------
#         self.detect = Detect(
#             nc=nc,
#             # anchors=detect_head.anchors,          # copy anchor config
#             ch=detect_head.channels              # input channels per FPN level
#         )

#         # ------------ Loss ----------------------------------------------
#         self.criterion = v8DetectionLoss(self.detect)

#     # -----------------------------------------------------------------
#     def forward(self, batch, compute_loss=True):
#         """
#         batch dicts produced by our dataloader.
#         Returns (pred, loss_dict) when compute_loss=True,
#                 pred otherwise.
#         """
#         if self.mode == "fusion":
#             # (1) split modalities ------------------------------------
#             rgb = batch["rgb"]["image"]            # [B,3,H,W]
#             nir = batch["nir"]["image"]            # [B,1,H,W]

#             feats_rgb = self.rgb_backbone_neck(rgb)   # list of 3 tensors
#             feats_nir = self.nir_backbone_neck(nir)

#             fused = []
#             for f_rgb, f_nir, shrink in zip(feats_rgb, feats_nir, self.shrink):
#                 f_cat = torch.cat([f_rgb, f_nir], 1)  # concatenate along C
#                 fused.append(shrink(f_cat))
#             preds = self.detect(fused)
#         else:
#             img = (batch["rgb"]["image"] if self.mode == "rgb"
#                    else batch["nir"]["image"])
#             feats = self.backbone_neck(img)
#             preds = self.detect(feats)

#         if not compute_loss:
#             return preds

#         targets = batch["rgb"]["boxes"], batch["rgb"]["labels"]  # any camera’s boxes—they’re same
#         loss, loss_items = self.criterion(preds, targets)
#         return preds, loss, loss_items

# # ------------- 4.  tiny training loop example -------------------------
# def train_loop(data_dir: str,
#                mode: str = "fusion",
#                epochs: int = 50,
#                batch_size: int = 8):

#     cams: List[Camera] = [Camera("rgb"), Camera("nir")]

#     train_ds = MultiCamDataset(data_dir, cams, SetType.TRAIN)
#     train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
#                           collate_fn=lambda x: x)  # keep list of dicts

#     model = YOLOv8RGBNIR(mode=mode,
#                          nc=len(train_ds.class_names)).to("cuda")

#     opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

#     for ep in range(1, epochs+1):
#         model.train()
#         tot = 0
#         for samples in train_dl:
#             # merge list→dict of stacked tensors so forward() is happy
#             merged: Dict[str, Dict] = {"rgb": {}, "nir": {}}
#             for k in ("image", "boxes", "labels"):
#                 merged["rgb"][k] = torch.stack(
#                     [s["rgb"][k] for s in samples]
#                 ).to("cuda")
#                 merged["nir"][k] = torch.stack(
#                     [s["nir"][k] for s in samples]
#                 ).to("cuda")

#             _, loss, _ = model(merged, compute_loss=True)

#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             opt.step()
#             tot += loss.item()

#         print(f"✅ epoch {ep:03d}  loss {tot/len(train_dl):.4f}")

# # ------------------ 5.  run ------------------------------------------
# if __name__ == "__main__":
#     data_root = "data"          # contains train.txt, classes.txt …
#     train_loop(data_root, mode="fusion")         # or 'rgb', 'nir'
