import math
import random
from typing import Tuple

import torchvision.transforms as T


class RandomErasing(object):
    def __init__(
        self,
        probability=0.5,
        sl=0.02,
        sh=0.4,
        r1=0.3,
        mean=(0.486, 0.459, 0.408),
    ):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img

        for _ in range(100):
            area = img.size(1) * img.size(2)
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size(2) and h < img.size(1):
                x1 = random.randint(0, img.size(1) - h)
                y1 = random.randint(0, img.size(2) - w)

                if img.size(0) == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]

                return img

        return img


def build_train_tf(image_size: Tuple[int, int], aug_cfg):
    H, W = image_size
    mean = aug_cfg["mean"]
    std = aug_cfg["std"]
    tf = []

    tf.append(T.Resize((H, W)))

    if aug_cfg.get("mirror", "none") == "random":
        tf.append(T.RandomHorizontalFlip(p=0.5))

    padding_cfg = aug_cfg.get("padding", {})
    crop_cfg = aug_cfg.get("random_crop", {})

    if padding_cfg.get("enabled", False):
        tf.append(T.Pad(int(padding_cfg.get("size", 10))))

    if crop_cfg.get("enabled", False):
        tf.append(T.RandomCrop((H, W)))

    tf.append(T.ToTensor())
    tf.append(T.Normalize(mean=mean, std=std))

    re_cfg = aug_cfg.get("random_erasing", {})
    if re_cfg.get("enabled", False):
        tf.append(
            RandomErasing(
                probability=float(re_cfg.get("probability", 0.5)),
                sl=float(re_cfg.get("sl", 0.02)),
                sh=float(re_cfg.get("sh", 0.4)),
                r1=float(re_cfg.get("r1", 0.3)),
                mean=tuple(re_cfg.get("mean", mean)),
            )
        )

    return T.Compose(tf)


def build_test_tf(image_size: Tuple[int, int], mean, std):
    H, W = image_size
    return T.Compose([
        T.Resize((H, W)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
