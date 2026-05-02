from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

import torch


TRAIN_LABEL_ATTR = "labels"
NUM_CLASSES_ATTR = "num_classes"

MARK_QUERY = 0
MARK_GALLERY = 1
MARK_MULTI_QUERY = 2
VALID_EVAL_MARKS = {MARK_QUERY, MARK_GALLERY, MARK_MULTI_QUERY}


@dataclass(frozen=True)
class ReIDTrainSample:
    """Canonical train sample: image tensor plus relabeled train identity."""

    image: torch.Tensor
    label: int

    def __iter__(self):
        yield self.image
        yield self.label


@dataclass(frozen=True, init=False)
class ReIDEvalSample:
    """Canonical eval sample: image tensor plus raw identity/camera metadata."""

    image: torch.Tensor
    pid: int
    camid: int
    image_name: str
    mark: int

    def __init__(
        self,
        image: torch.Tensor,
        pid: int,
        camid: int,
        image_name: str | None = None,
        mark: int | None = None,
        name: str | None = None,
    ):
        if image_name is None:
            image_name = name
        if image_name is None:
            raise TypeError("ReIDEvalSample requires image_name.")
        if mark is None:
            raise TypeError("ReIDEvalSample requires mark.")
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "pid", int(pid))
        object.__setattr__(self, "camid", int(camid))
        object.__setattr__(self, "image_name", str(image_name))
        object.__setattr__(self, "mark", int(mark))

    @property
    def name(self) -> str:
        return self.image_name

    def __iter__(self):
        yield self.image
        yield self.pid
        yield self.camid
        yield self.image_name
        yield self.mark


@runtime_checkable
class ReIDTrainDataset(Protocol):
    labels: Sequence[int]
    num_classes: int

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> ReIDTrainSample:
        ...


@runtime_checkable
class ReIDEvalDataset(Protocol):
    pids: Sequence[int]
    cams: Sequence[int]
    marks: Sequence[int]
    im_names: Sequence[str]

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> ReIDEvalSample:
        ...


def normalize_train_sample(sample: Any) -> ReIDTrainSample:
    if isinstance(sample, ReIDTrainSample):
        return sample
    if isinstance(sample, (tuple, list)) and len(sample) == 2:
        image, label = sample
        return ReIDTrainSample(image=image, label=int(label))
    raise TypeError(
        "Train samples must be ReIDTrainSample or a 2-tuple: "
        "(image, label)."
    )


def normalize_eval_sample(sample: Any) -> ReIDEvalSample:
    if isinstance(sample, ReIDEvalSample):
        return sample
    if isinstance(sample, (tuple, list)) and len(sample) == 5:
        image, pid, camid, name, mark = sample
        return ReIDEvalSample(
            image=image,
            pid=int(pid),
            camid=int(camid),
            image_name=str(name),
            mark=int(mark),
        )
    raise TypeError(
        "Eval samples must be ReIDEvalSample or a 5-tuple: "
        "(image, pid, camid, name, mark)."
    )


def validate_train_dataset(dataset: Any) -> None:
    missing = [
        attr for attr in (TRAIN_LABEL_ATTR, NUM_CLASSES_ATTR)
        if not hasattr(dataset, attr)
    ]
    if missing:
        raise TypeError(
            "Train datasets must expose labels and num_classes. "
            f"Missing: {', '.join(missing)}."
        )

    labels = list(getattr(dataset, TRAIN_LABEL_ATTR))
    if len(labels) != len(dataset):
        raise ValueError(
            f"Train dataset labels length must match dataset length, got "
            f"{len(labels)} labels for {len(dataset)} samples."
        )
    if not labels:
        raise ValueError("Train dataset is empty after protocol filtering.")

    num_classes = int(getattr(dataset, NUM_CLASSES_ATTR))
    if num_classes <= 0:
        raise ValueError(f"Train dataset num_classes must be positive, got {num_classes}.")
    if any(int(label) < 0 for label in labels):
        raise ValueError("Train dataset labels must be contiguous non-negative class labels.")
    label_set = {int(label) for label in labels}
    if max(label_set) >= num_classes:
        raise ValueError("Train dataset contains a label outside [0, num_classes).")
    expected_labels = set(range(num_classes))
    if label_set != expected_labels:
        raise ValueError(
            "Train dataset labels must be contiguous non-negative class labels "
            "covering [0, num_classes)."
        )


def validate_eval_dataset(dataset: Any) -> None:
    required = ("pids", "cams", "marks", "im_names")
    missing = [attr for attr in required if not hasattr(dataset, attr)]
    if missing:
        raise TypeError(
            "Eval datasets must expose pids, cams, marks, and im_names. "
            f"Missing: {', '.join(missing)}."
        )

    n = len(dataset)
    for attr in required:
        values = getattr(dataset, attr)
        if len(values) != n:
            raise ValueError(
                f"Eval dataset {attr} length must match dataset length, got "
                f"{len(values)} values for {n} samples."
            )
    if n == 0:
        raise ValueError("Eval dataset is empty.")

    marks = [int(mark) for mark in getattr(dataset, "marks")]
    invalid_marks = sorted({mark for mark in marks if mark not in VALID_EVAL_MARKS})
    if invalid_marks:
        raise ValueError(f"Eval dataset contains unsupported marks: {invalid_marks}.")
    if MARK_QUERY not in marks:
        raise ValueError("Eval dataset must contain at least one query sample (mark=0).")
    if MARK_GALLERY not in marks:
        raise ValueError("Eval dataset must contain at least one gallery sample (mark=1).")
