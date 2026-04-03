import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import numpy as np

from .models.maniqa import MANIQA_RQI
from .utils.io import load_image
from .hub import load_pretrained


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class RQI(nn.Module):
    def __init__(self, pretrained=True, device=None):
        super().__init__()

        setup_seed(20)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        self.device = device

        self.model = MANIQA_RQI().to(self.device)

        if pretrained:
            load_pretrained(self.model)

        self.model.eval()

        self.crop_size = 224
        self.downscale_factor = 2

    def random_crop(self, img1, img2, crop_size, n):
        _, h, w = img1.shape
        crops1, crops2 = [], []

        for _ in range(n):
            if h <= crop_size or w <= crop_size:
                crop1 = F.interpolate(img1.unsqueeze(0), size=(h, w),
                                      mode='bilinear', align_corners=False).squeeze(0)
                crop2 = F.interpolate(img2.unsqueeze(0), size=(h, w),
                                      mode='bilinear', align_corners=False).squeeze(0)
            else:
                x = random.randint(0, w - crop_size)
                y = random.randint(0, h - crop_size)
                crop1 = img1[:, y:y+crop_size, x:x+crop_size]
                crop2 = img2[:, y:y+crop_size, x:x+crop_size]

            crops1.append(crop1)
            crops2.append(crop2)

        return torch.stack(crops1), torch.stack(crops2)

    def forward(self, img1, img2):
        img1 = load_image(img1).to(self.device)
        img2 = load_image(img2).to(self.device)

        total_scores = 0
        total_count = 0
        h, w = img1.shape[1:]

        scale_patch_nums = [20, 10, 5]

        with torch.no_grad():
            for scale in range(3):
                n = scale_patch_nums[scale]

                crops1, crops2 = self.random_crop(img1, img2, self.crop_size, n)

                crops1 = crops1.to(self.device)
                crops2 = crops2.to(self.device)

                output = self.model(crops1, crops2)

                total_scores += output.sum()
                total_count += len(output)

                new_h, new_w = h // 2, w // 2
                if new_h < self.crop_size or new_w < self.crop_size:
                    break

                img1 = F.interpolate(img1.unsqueeze(0), size=(new_h, new_w),
                                     mode='bilinear', align_corners=False).squeeze(0)
                img2 = F.interpolate(img2.unsqueeze(0), size=(new_h, new_w),
                                     mode='bilinear', align_corners=False).squeeze(0)

                h, w = new_h, new_w

        final_score = total_scores / total_count if total_count > 0 else torch.tensor(0.0)

        return final_score.item()