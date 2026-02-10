# src/infrastructure/reid/osnet_loader.py
import torch
from torchreid.reid.utils import FeatureExtractor

class OSNetModelLoader:

    def __init__(self, device="cuda"):
        self.device = device
        self.extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            model_path="weights/osnet_x1_0_msmt17.pth",  # 또는 torchreid 다운로드 자동 사용
            device=device
        )

    def extract(self, img):
        # img: BGR ndarray from np/cv2
        feat = self.extractor(img)  # -> (1, 512)
        feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.squeeze(0)
