import torch
import torch.nn as nn
from collections import namedtuple

# Standard IR-SE implementation for ArcFace
# Based on TreB1eN/InsightFace_Pytorch and cftang0827

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class bottleneck_IR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
        )

    def forward(self, x):
        return self.shortcut_layer(x) + self.res_layer(x)


class bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False), 
                nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), 
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False), 
            nn.BatchNorm2d(depth), 
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class Backbone(nn.Module):
    def __init__(self, num_layers, drop_ratio, mode='ir_se'):
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152], "num_layers should be 50,100, or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = {
            50: [3, 4, 14, 3],
            100: [3, 13, 30, 3],
            152: [3, 8, 36, 3]
        }
        layers = blocks[num_layers]
        module_list = []
        in_channel = 64
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
            
        # Refactored for simpler sequential building
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False), 
            nn.BatchNorm2d(64), 
            nn.PReLU(64)
        )
        
        for i in range(4):
            module_list.append(self._make_layer(unit_module, in_channel, 64 * (2 ** i), layers[i], 2))
            in_channel = 64 * (2 ** i)
            
        self.body = nn.Sequential(*module_list)
        
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512), 
            nn.Dropout(drop_ratio), 
            Flatten(), 
            nn.Linear(512 * 7 * 7, 512), 
            nn.BatchNorm1d(512)
        )

    def _make_layer(self, block, in_channel, depth, blocks, stride):
        layers = []
        for i in range(blocks):
            if i == 0:
                layers.append(block(in_channel, depth, stride))
            else:
                layers.append(block(depth, depth, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x

def IR_SE_50(input_size):
    """
    Constructs a IR_SE_50 model.
    """
    model = Backbone(50, 0.6, 'ir_se')
    return model
