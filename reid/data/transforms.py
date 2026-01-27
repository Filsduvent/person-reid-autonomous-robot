from typing import Tuple
from torchvision import transforms as T

def build_train_tf(image_size: Tuple[int, int], mean, std, mirror: str, crop_prob: float, crop_ratio: float, scale_255: bool):
    H, W = image_size
    tf = []
    tf.append(T.Resize((H, W)))
    if crop_prob and crop_prob > 0 and crop_ratio and crop_ratio < 1.0:
        # simple approximation: RandomResizedCrop with limited scale
        tf.append(T.RandomApply([T.RandomResizedCrop((H, W), scale=(crop_ratio, 1.0))], p=crop_prob))
    if mirror == "random":
        tf.append(T.RandomHorizontalFlip(p=0.5))
    tf.append(T.ToTensor())  # outputs [0,1]
    # scale_255 in your YAML is conceptually handled already by ToTensor; keep flag for compatibility
    tf.append(T.Normalize(mean=mean, std=std))
    return T.Compose(tf)

def build_test_tf(image_size: Tuple[int, int], mean, std):
    H, W = image_size
    return T.Compose([
        T.Resize((H, W)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
