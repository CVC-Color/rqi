import cv2
import numpy as np
import torch
from PIL import Image as PILImage


def load_image(img):
    """
    input type:
    - str: image path
    - np.ndarray: cv2 / RGB 
    - torch.Tensor: (H,W,C) / (C,H,W) / (1,C,H,W)
    - PIL.Image
    """

    if isinstance(img, str):
        img_np = cv2.imread(img)
        if img_np is None:
            raise ValueError(f"Failed to read image from path: {img}")
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    elif isinstance(img, PILImage):
        img_np = np.array(img)

        if img_np.ndim == 2:
            pass
        elif img_np.ndim == 3:
            if img_np.shape[2] == 4:
                img_np = img_np[:, :, :3]
        else:
            raise ValueError(f"Invalid PIL image shape: {img_np.shape}")

    elif isinstance(img, np.ndarray):
        if img.ndim == 2:
            img_np = img
        elif img.ndim == 3:
            if img.shape[2] == 3:
                img_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 1:
                img_np = img.squeeze(2)
            elif img.shape[2] == 4:
                img_np = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                raise ValueError(f"Invalid numpy image shape: {img.shape}")
        else:
            raise ValueError(f"Unsupported numpy shape: {img.shape}")

    elif isinstance(img, torch.Tensor):
        img = img.clone().detach().float()

        if img.dim() == 4:
            if img.shape[0] == 1:
                img = img.squeeze(0)
            else:
                raise ValueError(f"Batch tensor not supported: {img.shape}")

        if img.dim() == 2:
            img = img.unsqueeze(0)

        if img.dim() == 3 and img.shape[0] not in [1, 3]:
            if img.shape[-1] in [1, 3]:
                img = img.permute(2, 0, 1)
            else:
                raise ValueError(f"Invalid tensor shape: {img.shape}")

        if img.shape[0] not in [1, 3]:
            raise ValueError(f"Invalid channel number: {img.shape}")

        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        if torch.isnan(img).any() or torch.isinf(img).any():
            raise ValueError("Tensor contains NaN or Inf")

        if img.max() > 1:
            img = img / 255.0

        img = (img - 0.5) / 0.5

        return img.contiguous()

    else:
        raise TypeError(f"Unsupported input type: {type(img)}")

    if np.isnan(img_np).any() or np.isinf(img_np).any():
        raise ValueError("Image contains NaN or Inf")

    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)

    if img_np.dtype != np.float32:
        img_np = img_np.astype("float32")

    if img_np.max() > 1:
        img_np = img_np / 255.0

    img_np = np.transpose(img_np, (2, 0, 1))

    img_np = (img_np - 0.5) / 0.5

    return torch.from_numpy(img_np).float().contiguous()