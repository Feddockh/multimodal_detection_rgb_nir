import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import Image, BoundingBoxes, BoundingBoxFormat
from typing import List, Dict, Tuple
from utils.camera import Camera


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class SetType:
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class MultiCamDataset(Dataset):
    def __init__(self, base_dir: str, cameras: List[Camera], set_type: SetType = SetType.TRAIN, transforms=None):
        self.base_dir = base_dir
        self.cameras = cameras
        self.set_type = set_type
        self.transforms = transforms

        # Load the image filenames from the set type file
        set_file = os.path.join(base_dir, f"{set_type}.txt")
        if not os.path.exists(set_file):
            raise FileNotFoundError(f"Set file {set_file} does not exist.")
        
        with open(set_file, "r") as f:
            self.image_filenames = [line.strip() for line in f.readlines()]
            self.image_filenames = sorted(self.image_filenames)

        # Load class names
        classes_file = os.path.join(base_dir, "classes.txt")
        if os.path.exists(classes_file):
            with open(classes_file, "r") as f:
                self.class_names = [line.strip() for line in f.readlines()]
        else:
            raise FileNotFoundError(f"Classes file \"{classes_file}\" does not exist.")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx) -> Dict[str, Tuple[Image, Dict[str, torch.Tensor], str]]:
        sample = {}
        img_filename = self.image_filenames[idx]

        for cam in self.cameras:
            cam_dir = os.path.join(self.base_dir, cam.name)
            img_path = os.path.join(cam_dir, "images", img_filename)
            label_path = os.path.join(cam_dir, "labels", os.path.splitext(img_filename)[0] + ".txt")

            # Load image
            img = read_image(img_path)  # [C, H, W], uint8
            img = F.to_dtype(img, dtype=torch.float32, scale=True)
            canvas_size = (img.shape[1], img.shape[2])  # (H, W)

            # Load YOLO boxes
            boxes = []
            labels = []

            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cls, xc, yc, w, h = map(float, parts)
                        labels.append(int(cls))
                        x = xc * canvas_size[1] - (w * canvas_size[1]) / 2
                        y = yc * canvas_size[0] - (h * canvas_size[0]) / 2
                        boxes.append([x, y, w * canvas_size[1], h * canvas_size[0]])

            if boxes:
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
                boxes_tv = BoundingBoxes(boxes_tensor, format=BoundingBoxFormat.XYWH, canvas_size=canvas_size)
                boxes_tv = F.convert_bounding_box_format(boxes_tv, new_format=BoundingBoxFormat.XYXY)
                labels_tensor = torch.tensor(labels, dtype=torch.int64)
            else:
                boxes_tv = BoundingBoxes(torch.empty((0, 4), dtype=torch.float32),
                                         format=BoundingBoxFormat.XYXY,
                                         canvas_size=canvas_size)
                labels_tensor = torch.empty((0,), dtype=torch.int64)

            target = {
                "boxes": boxes_tv,
                "labels": labels_tensor
            }

            if self.transforms:
                img, target = self.transforms(img, target)

            sample[cam.name] = (img, target)

        return self._filter_invalid_targets(sample)

    def _filter_invalid_targets(self, sample: Dict[str, Tuple[Image, Dict[str, torch.Tensor]]]):
        for cam in self.cameras:
            img, target = sample[cam.name]
            boxes = target['boxes']
            x1, y1, x2, y2 = boxes.unbind(dim=1)
            keep = (x2 > x1) & (y2 > y1)
            boxes = boxes[keep]
            labels = target['labels'][keep]
            sample[cam.name] = (img, {'boxes': boxes, 'labels': labels})
        return sample

    def get_class_names(self):
        return self.class_names


if __name__ == "__main__":
    # Example usage
    base_dir = "data"
    cameras = [Camera(name="firefly_left"), Camera(name="ximea_demosaic")]
    dataset = MultiCamDataset(base_dir, cameras)

    sample = dataset[0]
    import utils.visual as visual
    visual.plot([sample["firefly_left"]])
    visual.plot([sample["ximea_demosaic"]])