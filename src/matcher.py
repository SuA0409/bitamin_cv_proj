# from config import EVA_NAME, EVA_WEIGHT_NAME   # baseline
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue
from wildlife_tools.similarity.wildfusion import SimilarityPipeline
from wildlife_tools.features import DeepFeatures
from wildlife_tools.features.local import AlikedExtractor
from wildlife_tools.similarity.calibration import IsotonicCalibration
import torch
import torchvision

'''
MegaDescriptor, ALIKED matcher 각각 생성 + return
'''
# Windows 에러 해결을 위해 밖으로 꺼낸 클래스 및 함수들
class DinoV3Embedder(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        out = self.backbone(x)
        if isinstance(out, dict):
            return out["x_norm_clstoken"]
        return out

class EvaTransformWrapper:
    def __init__(self, preprocess):
        self.preprocess = preprocess
    def __call__(self, img, metadata=None):
        return self.preprocess(img)

class DinoTransformWrapper:
    def __init__(self, preprocess):
        self.preprocess = preprocess
    def __call__(self, img, metadata=None):
        return self.preprocess(img)

# MegaDescriptor matcher
def build_megadescriptor(model, transform, device='cuda', batch_size=16):
    return SimilarityPipeline(
        matcher=CosineSimilarity(),
        extractor=DeepFeatures(model=model, device=device, batch_size=batch_size),
        transform=transform,
        calibration=IsotonicCalibration()
    )

# ALIKED matcher
def build_aliked(transform, device='cuda', batch_size=16):
    return SimilarityPipeline(
        matcher=MatchLightGlue(features='aliked', device=device, batch_size=batch_size),
        extractor=AlikedExtractor(),
        transform=transform,
        calibration=IsotonicCalibration()
    )

# # EVA02-CLIP matcher
# try:
#     from open_clip import create_model_and_transforms
# except ImportError:
#     create_model_and_transforms = None

# def build_eva02(device='cuda', batch_size=16):
#     """
#     Global descriptor from EVA02-CLIP-L-14-336.
#     Returns SimilarityPipeline with Cosine similarity.
#     """
#     if create_model_and_transforms is None:
#         raise ImportError("open_clip_torch not installed. pip install open_clip_torch")

#     model, _, preprocess = create_model_and_transforms(
#         EVA_NAME,
#         pretrained=EVA_WEIGHT_NAME
#     )
#     model = model.visual.to(device).eval()

#     transform = EvaTransformWrapper(preprocess)

#     return SimilarityPipeline(
#         matcher=CosineSimilarity(),
#         extractor=DeepFeatures(model, device=device, batch_size=batch_size),
#         transform=transform,
#         calibration=IsotonicCalibration()
#     )

# # DinoV3 matcher
# def build_dinov3(device='cuda', batch_size=16):
#     # model load (torch.hub 사용해서 로드)
#     model = torch.hub.load("facebookresearch/dinov3", "dinov3_vits14")
#     model = DinoV3Embedder(model).to(device).eval()

#     preprocess = torchvision.transforms.Compose([
#         torchvision.transforms.Resize(256),
#         torchvision.transforms.CenterCrop(224),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                                          std=(0.229, 0.224, 0.225)),
#     ])

#     transform = DinoTransformWrapper(preprocess)

#     return SimilarityPipeline(
#         matcher=CosineSimilarity(),
#         extractor=DeepFeatures(model=model, device=device, batch_size=batch_size),
#         transform=transform,
#         calibration=IsotonicCalibration()
#     )
