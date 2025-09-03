# your_model.py
# Inference utils para Streamlit (contagem de "bolinhas")

from __future__ import annotations
import os
import io
import math
from typing import Tuple, Optional

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

# ==== Constantes de normalização (ImageNet) ====
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ==== Blocos da rede (replicados do notebook) ====

class DDCB(nn.Module):
    def __init__(self, in_planes: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, 256, 1, bias=False), nn.ReLU(True),
            nn.Conv2d(256, 64, 3, padding=1, bias=False), nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_planes+64, 256, 1, bias=False), nn.ReLU(True),
            nn.Conv2d(256, 64, 3, padding=2, dilation=2, bias=False), nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_planes+64*2, 256, 1, bias=False), nn.ReLU(True),
            nn.Conv2d(256, 64, 3, padding=3, dilation=3, bias=False), nn.ReLU(True)
        )
        self.conv4 = nn.Conv2d(in_planes+64*3, 512, 3, padding=1, bias=False)

    def forward(self, x):
        x1_raw = self.conv1(x)
        x1 = torch.cat([x, x1_raw], dim=1)
        x2_raw = self.conv2(x1)
        x2 = torch.cat([x, x1_raw, x2_raw], dim=1)
        x3_raw = self.conv3(x2)
        x3 = torch.cat([x, x1_raw, x2_raw, x3_raw], dim=1)
        return self.conv4(x3)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class DenseScaleNet(nn.Module):
    def __init__(self, load_model: str = ''):
        super().__init__()
        self.load_model = load_model
        self.features_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.features = make_layers(self.features_cfg)
        for p in self.features.parameters():
            p.requires_grad = False
        self.DDCB1 = DDCB(512)
        self.DDCB2 = DDCB(512)
        self.DDCB3 = DDCB(512)
        self.output_layers = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 1, 1)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x1_raw = self.DDCB1(x)
        x1 = x1_raw + x
        x2_raw = self.DDCB2(x1)
        x2 = x2_raw + x1_raw + x
        x3_raw = self.DDCB3(x2)
        x3 = x3_raw + x2_raw + x1_raw + x
        return self.output_layers(x3)

    def _random_initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _initialize_weights(self):
        self_dict = self.state_dict()
        self._random_initialize_weights()
        # Para inferência, não precisamos baixar VGG. Apenas mantemos a estrutura.
        # (No treino original, havia injeção de VGG16; para carregar ckpt isso não é necessário.)


# ==== Utilidades de inferência ====

def _normalize_imagenet(img_rgb: np.ndarray) -> np.ndarray:
    """img_rgb: HxWx3 em [0..255]"""
    img = img_rgb.astype(np.float32) / 255.0
    img = (img - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return img

def _cm_jet(gray: np.ndarray) -> np.ndarray:
    """gray: [0..1] -> RGB uint8"""
    gray8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    color = cv2.applyColorMap(gray8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return color

def _to_multiple_of_8(h: int, w: int) -> Tuple[int, int]:
    newH = (h + 7) // 8 * 8
    newW = (w + 7) // 8 * 8
    return newH, newW


def load_model(ckpt_path: str, device: str = "cpu") -> Tuple[nn.Module, torch.device]:
    """
    Carrega a DenseScaleNet e aplica o state_dict do seu checkpoint.
    Retorna (model.eval(), device).
    """
    dev = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = DenseScaleNet().to(dev)
    # Carregador compatível com versões novas/antigas do PyTorch
    import inspect
    kw = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kw["weights_only"] = False
    sd = torch.load(ckpt_path, **kw)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.eval()
    model.to(dev)
    return model, dev


def infer_image_pil(
    image: Image.Image,
    model: nn.Module,
    device: torch.device,
    downsample: int = 2,
    return_overlay: bool = True,
) -> Tuple[float, Optional[Image.Image]]:
    """
    Recebe uma PIL.Image, retorna (count, overlay opcional).
    """
    # PIL -> RGB np.uint8
    img_rgb = np.array(image.convert("RGB"))
    H, W = img_rgb.shape[:2]
    newH, newW = _to_multiple_of_8(H, W)

    # resize p/ múltiplo de 8
    if (newH, newW) != (H, W):
        img_res = cv2.resize(img_rgb, (newW, newH), interpolation=cv2.INTER_LINEAR)
    else:
        img_res = img_rgb

    inp = _normalize_imagenet(img_res)
    t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = model(t)  # 1x1xhxw (em feature scale)
        # Reamostragem para o grid de densidade esperado (downsample do treino)
        pred = F.interpolate(
            pred, size=(newH // downsample, newW // downsample),
            mode="bilinear", align_corners=False
        )
        count = float(pred.sum().cpu().item())

    overlay_img = None
    if return_overlay:
        dm = pred.squeeze().cpu().numpy()
        dm = dm / (dm.max() + 1e-8)
        dm_up = cv2.resize(dm, (newW, newH), interpolation=cv2.INTER_LINEAR)
        heat = (_cm_jet(dm_up)).astype(np.uint8)
        overlay = cv2.addWeighted(cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR), 0.5,
                                  cv2.cvtColor(heat, cv2.COLOR_RGB2BGR), 0.5, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        cv2.putText(overlay, f"count={count:.1f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2, cv2.LINE_AA)
        overlay_img = Image.fromarray(overlay)

    return count, overlay_img


def infer_image_path(
    image_path: str,
    model: nn.Module,
    device: torch.device,
    downsample: int = 2,
    return_overlay: bool = True,
) -> Tuple[float, Optional[Image.Image]]:
    img = Image.open(image_path).convert("RGB")
    return infer_image_pil(img, model, device, downsample=downsample, return_overlay=return_overlay)


# Opcional: função de pasta (útil para debug local)
def infer_folder(
    ckpt_path: str,
    images_dir: str,
    out_dir: str,
    downsample: int = 2,
    device: str = "cpu",
    write_vis: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)
    model, dev = load_model(ckpt_path, device=device)

    # coleta imagens
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    paths = []
    for root, _, files in os.walk(images_dir):
        for nm in files:
            if nm.lower().endswith(exts):
                paths.append(os.path.join(root, nm))
    paths.sort()

    rows = ["filename,count"]
    for p in paths:
        cnt, ov = infer_image_path(p, model, dev, downsample=downsample, return_overlay=write_vis)
        rows.append(f"{os.path.basename(p)},{cnt:.1f}")
        if write_vis and ov is not None:
            ov.save(os.path.join(out_dir, os.path.basename(p)))

    with open(os.path.join(out_dir, "counts.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows))


if __name__ == "__main__":
    # Exemplo rápido: python your_model.py /caminho/modelo.pth /pasta/imagens /pasta/saida
    import sys
    if len(sys.argv) >= 4:
        ckpt, imgdir, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
        infer_folder(ckpt, imgdir, outdir, downsample=2, device="cuda" if torch.cuda.is_available() else "cpu")
        print("Pronto.")
    else:
        print("Uso: python your_model.py <ckpt.pth> <images_dir> <out_dir>")
