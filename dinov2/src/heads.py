# %%
import torch
import torch.nn as nn


class LinearClassifierToken(torch.nn.Module):
    def __init__(self, in_channels, tokenW=16, tokenH=16, num_labels=1):
        super(LinearClassifierToken, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1, 1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

        return self.classifier(embeddings)


class UNetDecoderUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim=1024) -> None:
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.skip_conv = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        x2 = self.skip_conv(x2)
        scale_factor = x1.size()[2] / x2.size()[2]
        x2 = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)(x2)
        x = torch.concat([x1, x2], dim=1)
        return self.conv(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
                mode = "fan_out" if isinstance(m, torch.nn.Conv2d) else "fan_in"
                torch.nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity="relu")
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)


class UNetDecoder(nn.Module):
    """Unet decoder head"""

    DECODER_TYPE = "unet"

    def __init__(self, in_channels, out_channels, image_size=224, patch_size=14):
        super(UNetDecoder, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = in_channels
        self.image_size = image_size
        self.up1 = UNetDecoderUpBlock(
            in_channels=in_channels, out_channels=in_channels // 4, embed_dim=self.embed_dim
        )
        self.up2 = UNetDecoderUpBlock(
            in_channels=in_channels // 4, out_channels=in_channels // 4, embed_dim=self.embed_dim
        )
        self.up3 = UNetDecoderUpBlock(
            in_channels=in_channels // 4, out_channels=in_channels // 4, embed_dim=self.embed_dim
        )
        self.up4 = UNetDecoderUpBlock(
            in_channels=in_channels // 4, out_channels=out_channels, embed_dim=self.embed_dim
        )

    def forward(self, x):

        h = w = self.image_size // self.patch_size

        skip1 = x[3].reshape(-1, h, w, self.embed_dim).permute(0, 3, 1, 2)
        skip2 = x[2].reshape(-1, h, w, self.embed_dim).permute(0, 3, 1, 2)
        skip3 = x[1].reshape(-1, h, w, self.embed_dim).permute(0, 3, 1, 2)
        skip4 = x[0].reshape(-1, h, w, self.embed_dim).permute(0, 3, 1, 2)
        x1 = x[4].reshape(-1, h, w, self.embed_dim).permute(0, 3, 1, 2)

        x2 = self.up1(x1, skip1)
        x3 = self.up2(x2, skip2)
        x4 = self.up3(x3, skip3)
        x5 = self.up4(x4, skip4)

        return x5
