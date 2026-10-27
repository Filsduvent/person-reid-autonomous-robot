import copy

import torch
import torchvision.transforms as T
from PIL import Image

from reid.data.transforms import RandomErasing, build_test_tf, build_train_tf


BASE_AUG_CFG = {
    "scale_255": True,
    "mean": [0.486, 0.459, 0.408],
    "std": [0.229, 0.224, 0.225],
    "mirror": "random",
    "padding": {
        "enabled": True,
        "size": 10,
    },
    "random_crop": {
        "enabled": True,
    },
    "random_erasing": {
        "enabled": True,
        "probability": 0.5,
        "sl": 0.02,
        "sh": 0.4,
        "r1": 0.3,
        "mean": [0.486, 0.459, 0.408],
    },
}


def test_train_transform_contains_random_erasing_only_when_enabled():
    aug_cfg = copy.deepcopy(BASE_AUG_CFG)
    tf = build_train_tf(image_size=(256, 128), aug_cfg=aug_cfg)

    assert any(isinstance(op, RandomErasing) for op in tf.transforms)

    aug_cfg["random_erasing"]["enabled"] = False
    tf_disabled = build_train_tf(image_size=(256, 128), aug_cfg=aug_cfg)

    assert not any(isinstance(op, RandomErasing) for op in tf_disabled.transforms)


def test_train_transform_uses_locked_preprocessing_order():
    tf = build_train_tf(image_size=(256, 128), aug_cfg=copy.deepcopy(BASE_AUG_CFG))

    assert [type(op) for op in tf.transforms] == [
        T.Resize,
        T.RandomHorizontalFlip,
        T.Pad,
        T.RandomCrop,
        T.ToTensor,
        T.Normalize,
        RandomErasing,
    ]


def test_test_transform_never_contains_random_erasing():
    tf = build_test_tf(
        image_size=(256, 128),
        mean=BASE_AUG_CFG["mean"],
        std=BASE_AUG_CFG["std"],
    )

    assert not any(isinstance(op, RandomErasing) for op in tf.transforms)


def test_test_transform_uses_eval_only_preprocessing_order():
    tf = build_test_tf(
        image_size=(256, 128),
        mean=BASE_AUG_CFG["mean"],
        std=BASE_AUG_CFG["std"],
    )

    assert [type(op) for op in tf.transforms] == [
        T.Resize,
        T.ToTensor,
        T.Normalize,
    ]


def test_train_transform_outputs_tensor_of_expected_shape():
    aug_cfg = copy.deepcopy(BASE_AUG_CFG)
    aug_cfg["random_erasing"]["enabled"] = False
    tf = build_train_tf(image_size=(256, 128), aug_cfg=aug_cfg)

    img = Image.new("RGB", (64, 32), color=(255, 255, 255))
    out = tf(img)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 256, 128)


def test_explicit_bicubic_matches_pil_pixels_and_omitted_resize_is_unchanged():
    import numpy as np
    image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (17, 9, 3), dtype=np.uint8))
    mean, std = [0., 0., 0.], [1., 1., 1.]
    expected = T.ToTensor()(image.resize((5, 11), resample=Image.Resampling.BICUBIC))
    test_tf = build_test_tf((11, 5), mean, std, interpolation='bicubic')
    torch.testing.assert_close(test_tf(image), expected, rtol=0, atol=0)
    aug = dict(mean=mean, std=std, mirror='none')
    torch.testing.assert_close(build_train_tf((11, 5), aug, interpolation='bicubic')(image), expected, rtol=0, atol=0)
    historical = T.Compose([T.Resize((11, 5)), T.ToTensor(), T.Normalize(mean, std)])(image)
    torch.testing.assert_close(build_test_tf((11, 5), mean, std)(image), historical, rtol=0, atol=0)
    torch.testing.assert_close(build_train_tf((11, 5), aug)(image), historical, rtol=0, atol=0)


def test_invalid_interpolation_fails_transform_and_config_validation():
    import pytest
    from reid.utils.config import validate_reid_config
    for value in ['unknown', 3, True]:
        with pytest.raises(ValueError, match='interpolation'):
            build_test_tf((11, 5), [0]*3, [1]*3, interpolation=value)
        with pytest.raises(ValueError, match='interpolation'):
            validate_reid_config({'data': {'train': {'images': {'interpolation': value}}}})
