import torch
import torch.nn as nn
from facenn.models.base import FaceRecognitionModel
from facenn.config import Config, DEVICE
from facenn.utils.io import download_file_from_url
import os

# MobileFaceNet v2 implementation (based on MobileNetV2 style with Face recognition tweaks)
class ConvBN(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, padding, groups=1):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class LinearBottleneck(nn.Module):
    def __init__(self, in_c, out_c, stride, expand_ratio):
        super(LinearBottleneck, self).__init__()
        self.stride = stride
        hidden_dim = int(in_c * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_c == out_c

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBN(in_c, hidden_dim, 1, 1, 0))
        layers.extend([
            ConvBN(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileFaceNetV2Model(nn.Module):
    def __init__(self, embedding_size=512):
        super(MobileFaceNetV2Model, self).__init__()
        self.layers = nn.Sequential(
            ConvBN(3, 32, 3, 2, 1),
            LinearBottleneck(32, 16, 1, 1),
            LinearBottleneck(16, 24, 2, 6),
            LinearBottleneck(24, 24, 1, 6),
            LinearBottleneck(24, 32, 2, 6),
            LinearBottleneck(32, 32, 1, 6),
            LinearBottleneck(32, 32, 1, 6),
            LinearBottleneck(32, 64, 2, 6),
            LinearBottleneck(64, 64, 1, 6),
            LinearBottleneck(64, 64, 1, 6),
            LinearBottleneck(64, 64, 1, 6),
            LinearBottleneck(64, 96, 1, 6),
            LinearBottleneck(96, 96, 1, 6),
            LinearBottleneck(96, 96, 1, 6),
            LinearBottleneck(96, 160, 2, 6),
            LinearBottleneck(160, 160, 1, 6),
            LinearBottleneck(160, 160, 1, 6),
            LinearBottleneck(160, 320, 1, 1),
            ConvBN(320, 1280, 1, 1, 0),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(1280, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.layers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.bn(x)
        return x

class MobileFaceNetV2(FaceRecognitionModel):
    def __init__(self):
        super().__init__(model_name="MobileFaceNet v2", input_shape=(112, 112))
        
    def load_model(self):
        self.model = MobileFaceNetV2Model(embedding_size=512)
        url = "https://huggingface.co/Anirban_Dhar/Face_Recognition/resolve/main/Model_Training/MobileFaceNet_Pytorch/weights/mobilefacenet.pth"
        weights_path = Config.get_weights_path("MobileFaceNetV2", "mobilefacenet_v2.pth")

        if not os.path.exists(weights_path):
            download_file_from_url(url, weights_path)

        state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(DEVICE)
        self.model.eval()

    def forward(self, x):
        return self.model(x)
