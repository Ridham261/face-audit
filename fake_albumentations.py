"""
Minimal albumentations shim for glasses-detector inference.
Replaces albumentations with torchvision-based ImageNet normalize + to-tensor.
Injected via sys.modules before importing glasses-detector.
"""
import types as _types
import torch as _torch
import numpy as _numpy

_MEAN = _torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = _torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class _Stub:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, image=None, **kw):
        return {"image": image, **kw}


class Normalize(_Stub):
    """ImageNet-stats normalize: uint8 HWC → float CHW tensor."""
    def __call__(self, image=None, **kw):
        if isinstance(image, _torch.Tensor):
            return {"image": image, **kw}
        t = _torch.from_numpy(_numpy.asarray(image, dtype=_numpy.float32)).permute(2, 0, 1) / 255.0
        return {"image": (t - _MEAN) / _STD, **kw}


# Training-only stubs — instantiated but never called at inference time
VerticalFlip = _Stub
HorizontalFlip = _Stub
RandomRotate90 = _Stub
ShiftScaleRotate = _Stub
RandomResizedCrop = _Stub
GridDistortion = _Stub
OpticalDistortion = _Stub
PiecewiseAffine = _Stub
Perspective = _Stub
RandomBrightnessContrast = _Stub
RandomGamma = _Stub
CLAHE = _Stub
ColorJitter = _Stub
HueSaturationValue = _Stub
Blur = _Stub
GaussianBlur = _Stub
MedianBlur = _Stub
GaussNoise = _Stub
CoarseDropout = _Stub


class OneOf(_Stub):
    def __init__(self, transforms, p=0.5, **kw): pass


class Compose:
    def __init__(self, transforms, *args, **kw):
        self.transforms = transforms

    def __call__(self, image=None, **kw):
        result = {"image": image, **kw}
        for t in self.transforms:
            out = t(**result)
            result.update(out)
        return result


class _ToTensorV2(_Stub):
    def __call__(self, image=None, **kw):
        if isinstance(image, _torch.Tensor):
            return {"image": image, **kw}
        t = _torch.from_numpy(_numpy.asarray(image)).permute(2, 0, 1).float() / 255.0
        return {"image": t, **kw}


# Build the pytorch sub-module that glasses-detector imports from
pytorch = _types.ModuleType("albumentations.pytorch")
pytorch.ToTensorV2 = _ToTensorV2
