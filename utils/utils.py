import torch
from torchvision.transforms import transforms
from . import transforms_video


def build_transforms():
    mean = [124 / 255, 117 / 255, 104 / 255]
    std = [1 / (.0167 * 255)] * 3
    resize = 300, 224
    crop = 224

    res = transforms.Compose([
        transforms_video.ToTensorVideo(),
        transforms_video.ResizeVideo(resize),
        transforms_video.CenterCropVideo(crop),
        transforms_video.NormalizeVideo(mean=mean, std=std)
    ])

    return res
