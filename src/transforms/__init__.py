from .transform import (
    pixelbert_transform,
    pixelbert_transform_randaug,
    vit_transform,
    vit_transform_randaug,
    imagenet_transform,
    imagenet_transform_randaug,
    clip_transform,
    clip_transform_randaug,
    blip_transform,
    blip_transform_randaug,
    blip_transform_randaug_wc,
    blip_transform_randaug_wohf,
    blip_transform_randaug_pretrain,
)

_transforms = {
    "pixelbert": pixelbert_transform,
    "pixelbert_randaug": pixelbert_transform_randaug,
    "vit": vit_transform,
    "vit_randaug": vit_transform_randaug,
    "imagenet": imagenet_transform,
    "imagenet_randaug": imagenet_transform_randaug,
    "clip": clip_transform,
    "clip_randaug": clip_transform_randaug,
    "blip": blip_transform,
    "blip_randaug": blip_transform_randaug,
    "blip_randaug_wc": blip_transform_randaug_wc,
    "blip_randaug_wohf": blip_transform_randaug_wohf,
    "blip_randaug_pretrain": blip_transform_randaug_pretrain,
}

def keys_to_transforms(keys: list, size=224):
    return [_transforms[key](size=size) for key in keys]
