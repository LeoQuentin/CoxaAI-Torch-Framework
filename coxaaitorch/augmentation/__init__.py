from coxaaitorch.augmentation.autoaugment import (
    ImageNetPolicy,
    CIFAR10Policy,
    SVHNPolicy,
)

from coxaaitorch.augmentation.transforms import (
    no_augmentation,
    light_augmentation,
    autoaugment_policy_augmentation,
    random_augmentation,
)

__all__ = [
    "ImageNetPolicy",
    "CIFAR10Policy",
    "SVHNPolicy",
    "no_augmentation",
    "light_augmentation",
    "autoaugment_policy_augmentation",
    "random_augmentation",
]
