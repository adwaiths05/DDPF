import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import config as cfg

class SwinTinyBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=cfg.IMAGE_SIZE,
        )

    def forward(self, x):
        features = self.backbone(x)
        outputs = []
        for f in features:
            if f.shape[-1] in [96, 192, 384, 768]:
                f = f.permute(0, 3, 1, 2).contiguous()
            outputs.append(f)
        return outputs

def norm_layer(channels):
    groups = 32
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, channels)

class ConvGNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            norm_layer(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvGNAct(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            norm_layer(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x + identity)
        return x

class SwinUNetDecoder(nn.Module):
    def __init__(self, in_channels=[96, 192, 384, 768], out_channels=256):
        super().__init__()
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = ConvGNAct(in_channels[3] + in_channels[2], in_channels[2])
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = ConvGNAct(in_channels[2] + in_channels[1], in_channels[1])
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = ConvGNAct(in_channels[1] + in_channels[0], out_channels)
        self.hires_stem = nn.Sequential(
            ConvGNAct(3, 32),
            ConvGNAct(32, 32),
        )
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv4 = ConvGNAct(out_channels + 32, out_channels)
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv5 = ConvGNAct(out_channels + 32, out_channels)
        self.shared_conv = ResidualBlock(out_channels)

    def forward(self, c1, c2, c3, c4, image):
        x = self.up1(c4)
        x = torch.cat([x, c3], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, c2], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, c1], dim=1)
        x = self.conv3(x)
        hires = self.hires_stem(image)
        hires_half = F.avg_pool2d(hires, 2)
        x = self.up4(x)
        x = torch.cat([x, hires_half], dim=1)
        x = self.conv4(x)
        x = self.up5(x)
        x = torch.cat([x, hires], dim=1)
        x = self.conv5(x)
        return self.shared_conv(x)

class DensityHead(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.head = nn.Sequential(
            ConvGNAct(in_channels, 128),
            ConvGNAct(128, 64),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
        nn.init.constant_(self.head[-1].bias, -4.0)

    def forward(self, x):
        return F.softplus(self.head(x))

class LocalizationHead(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.head = nn.Sequential(
            ConvGNAct(in_channels, 128),
            ConvGNAct(128, 64),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
        nn.init.constant_(self.head[-1].bias, -2.19)

    def forward(self, x):
        return self.head(x)

class OffsetHead(nn.Module):
    def __init__(self, in_channels, out_channels=2):
        super().__init__()
        self.head = nn.Sequential(
            ConvGNAct(in_channels, 128),
            ConvGNAct(128, 64),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
        nn.init.constant_(self.head[-1].bias, 0.0)

    def forward(self, x):
        return 0.5 * torch.tanh(self.head(x))

class DDPFNet(nn.Module):
    def freeze_early_stages(self):
        for name, param in self.encoder.named_parameters():
            if any(name.startswith(p) for p in ["patch_embed", "layers.0", "layers.1"]):
                param.requires_grad = False

    def __init__(self, pretrained=True, shared_channels=256):
        super().__init__()
        self.encoder = SwinTinyBackbone(pretrained=pretrained)
        self.decoder = SwinUNetDecoder(in_channels=[96, 192, 384, 768], out_channels=shared_channels)
        self.density_head = DensityHead(in_channels=shared_channels)
        self.heatmap_head = LocalizationHead(in_channels=shared_channels)
        self.offset_head = OffsetHead(in_channels=shared_channels)

    def forward(self, x):
        c1, c2, c3, c4 = self.encoder(x)
        shared_feat = self.decoder(c1, c2, c3, c4, x)
        density = self.density_head(shared_feat)
        heatmap_logits = self.heatmap_head(shared_feat)
        offset = self.offset_head(shared_feat)
        return {
            "density": density,
            "heatmap_logits": heatmap_logits,
            "offset": offset
        }

DDPFNetV3 = DDPFNet
