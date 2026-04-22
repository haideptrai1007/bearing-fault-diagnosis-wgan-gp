import torch
from torchvision.transforms import v2
import torch.nn as nn
from torchvision.utils import make_grid
from torch.nn import functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import gc
from torchvision.models import inception_v3, Inception_V3_Weights
import csv


class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
    def forward(self, x):
        return self.main(x) + self.upsample(x) + self.shortcut(x)

class ResidualDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return self.main(x) + self.shortcut(x)
        

class Generator(nn.Module):
    def __init__(self, latent_dim=256):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.project = nn.Sequential(
            nn.Linear(latent_dim, 1024 * 4 * 4), 
            nn.ReLU(inplace=True)
        )
        self.main = nn.Sequential(
            # 8x8
            UpsampleConv(1024, 512),
            # 16x16
            UpsampleConv(512, 256), 
            # 32x32
            UpsampleConv(256, 256),
            # 64x64
            UpsampleConv(256, 256), 
            # 128x128
            UpsampleConv(256, 256),
            nn.Conv2d(256, 1, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.project(x)
        x = x.view(-1, 1024, 4, 4)
        x = self.main(x)
        return x
    
    def sample_latent(self, num_samples, device="cpu"):
        return torch.randn((num_samples, self.latent_dim), device=device)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 64x64
            ResidualDownsample(in_channels, 128, 4, 2, 1),
            # 32x32
            ResidualDownsample(128, 256, 4, 2, 1),
            # 16x16
            ResidualDownsample(256, 512, 4, 2, 1),
            # 8x8
            ResidualDownsample(512, 512, 4, 2, 1),
            # 4x4
            ResidualDownsample(512, 512, 4, 2, 1)
        )
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1)
        )
    
    def forward(self, x):
        x = self.main(x)
        x = self.output(x)
        return x
    
class FeatureExtractor:
    """Extracts pool-3 (2048-d) features from InceptionV3."""

    def __init__(self, device="cpu"):
        self.device = device
        self.model = None  # lazy-loaded

    def _load(self):
        if self.model is None:
            model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
            model.fc = nn.Identity()       # pool3 → 2048-d
            model.eval()
            self.model = model

    def _to_inception_input(self, imgs):
        """Convert 1-ch [-1,1] 128×128 → 3-ch [0,1] 299×299."""
        # expand to 3 channels
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        # scale from [-1,1] to [0,1]
        imgs = imgs * 0.5 + 0.5
        # resize to 299×299
        imgs = F.interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False)
        # ImageNet normalisation
        mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 3, 1, 1)
        imgs = (imgs - mean) / std
        return imgs

    @torch.no_grad()
    def extract(self, images, batch_size=16):
        """
        images : Tensor (N, 1, 128, 128) in [-1, 1]
        Returns: numpy array (N, 2048)
        """
        self._load()
        self.model.to(self.device)

        feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(self.device)
            batch = self._to_inception_input(batch)
            f = self.model(batch)
            feats.append(f.cpu())
            del batch, f
            torch.cuda.empty_cache()

        self.model.cpu()          # free VRAM immediately
        torch.cuda.empty_cache()
        return torch.cat(feats, dim=0).numpy()

    def cleanup(self):
        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()