import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from ultralytics.data import YOLODataset
from ultralytics.utils.

class PairedCameraDataset(Dataset):
    """
    Left side: full YOLODataset (with labels).
    Right side: just raw images, letter‐boxed to the same size.
    """
    def __init__(
        self,
        left_ds: YOLODataset,
        right_folder: str,
        exts=("*.png","*.jpg","*.jpeg"),
    ):
        self.left_ds = left_ds
        self.imgsz  = left_ds.imgsz  # e.g. 640

        # build stem→left‐index map
        self.map_l = {Path(p).stem: i for i, p in enumerate(left_ds.im_files)}

        # gather all right‐view files with allowed extensions
        files_r = []
        p2 = Path(right_folder)
        for e in exts:
            files_r += list(p2.rglob(e))
        map_r = {p.stem: p for p in files_r}

        # intersect stems & sort
        common = sorted(set(self.map_l) & set(map_r.keys()))
        if not common:
            raise RuntimeError("No overlapping frames!")

        # aligned indices + paths
        self.left_idx  = [self.map_l[s] for s in common]
        self.right_paths = [map_r[s] for s in common]

    def __len__(self):
        return len(self.left_idx)

    def __getitem__(self, idx):
        # 1) pull the left sample (img, labels, batch_idx, etc.)
        sample = self.left_ds[self.left_idx[idx]]

        # 2) load & letterbox the right image
        im2 = cv2.imread(str(self.right_paths[idx]))[:, :, ::-1]  # BGR→RGB
        im2 = LetterBox(im2).new_shape
        # to tensor, normalize 0–1, reorder channels
        im2 = torch.from_numpy(im2).permute(2, 0, 1).to(sample["img"].dtype) / 255.0

        # 3) stick it into the same dict as "img2"
        sample["img2"] = im2
        return sample

    @staticmethod
    def collate_fn(batch):
        # 1) stack all img2
        imgs2 = torch.stack([b["img2"] for b in batch], dim=0)
        # 2) strip img2 before default collation
        stripped = [{k: v for k, v in b.items() if k != "img2"} for b in batch]
        coll = YOLODataset.collate_fn(stripped)
        # 3) re-attach
        coll["img2"] = imgs2
        return coll

# ─── quick demo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) make your left‐only YOLODataset (with labels)
    data_cfg = {"names": {0: "firefly_off", 1: "firefly_on"}, "channels": 3}
    ds_left = YOLODataset(
        img_path="data/firefly_left/images",
        data=data_cfg, task="detect", augment=False,
    )

    # 2) wrap it with the paired camera dataset
    paired = PairedCameraDataset(ds_left, right_folder="data/ximea/images")

    # 3) load a batch
    loader = DataLoader(paired, batch_size=2, shuffle=False, num_workers=0,
                        collate_fn=PairedCameraDataset.collate_fn)
    batch = next(iter(loader))

    print(" left imgs:", batch["img"].shape)   # (B, 3, 640, 640)
    print("right imgs:", batch["img2"].shape)  # (B, 3, 640, 640)
    print("   boxes:", batch["bboxes"].shape)  # detection targets from left view
    print("    cls :", batch["cls"].shape)
    print("batch_idx:", batch["batch_idx"])