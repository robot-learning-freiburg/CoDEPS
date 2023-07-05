import operator
from typing import List

from datasets.cityscapes import Cityscapes
from datasets.cityscapes_labels import Label
from datasets.cityscapes_labels import labels as cityscapes_labels
from datasets.codeps_labels import labels as codeps_labels
from datasets.kitti_360 import Kitti360
from datasets.sem_kitti_dvps import SemKittiDvps

__all__ = ["Cityscapes", "Kitti360", "SemKittiDvps"]


def get_labels(remove_classes: List[int], mode: str) -> List[Label]:
    if mode == "cityscapes":
        labels = [label for label in cityscapes_labels if label.trainId not in [-1, 255]]
    elif mode == "codeps":
        labels = [label for label in codeps_labels if label.trainId not in [-1, 255]]
    else:
        raise ValueError(f"Unsupported label mode: {mode}")
    labels = sorted(labels, key=operator.attrgetter("trainId"))

    train_id = 0
    adapted_labels = []
    for label in labels:
        if label.trainId in remove_classes:
            continue
        label = label._replace(trainId=train_id)
        train_id += 1
        adapted_labels.append(label)

    return adapted_labels
