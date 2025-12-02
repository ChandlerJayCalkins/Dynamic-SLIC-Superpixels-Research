import os
import cv2
from pycocotools.coco import COCO

class COCODataset:
    def __init__(self, images_dir, annotation_file):
        self.images_dir = images_dir
        self.coco = COCO(annotation_file)
        self.image_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.image_ids)

    def get_image_info(self, idx):
        img_id = self.image_ids[idx]
        return self.coco.loadImgs(img_id)[0]

    def load_image_bgr(self, idx):
        info = self.get_image_info(idx)
        path = os.path.join(self.images_dir, info["file_name"])
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return img, info

    def load_image_rgb(self, idx):
        bgr, info = self.load_image_bgr(idx)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, info
