from torch import nn
import torch


class ResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.convolution1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.normalization1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.convolution2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.normalization2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs):
        residual = self.shortcut(inputs)
        outputs = self.activation(self.normalization1(self.convolution1(inputs)))
        outputs = self.normalization2(self.convolution2(outputs))
        return self.activation(outputs + residual)


class ResNetResidualAdd(nn.Module):
    """Standalone residual add with the same tensor shape on both branches."""

    def forward(self, inputs):
        return inputs + inputs


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, blocks=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    @staticmethod
    def _make_layer(in_channels, out_channels, blocks, stride=1):
        layers = [ResNetBasicBlock(in_channels, out_channels, stride)]
        layers.extend(
            ResNetBasicBlock(out_channels, out_channels) for _ in range(blocks - 1)
        )
        return nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = self.stem(inputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.pool(outputs)
        return self.classifier(torch.flatten(outputs, 1))