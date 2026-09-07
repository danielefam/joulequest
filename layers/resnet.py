from torch import nn
import torch


class ResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels if mid_channels is not None else out_channels
        self.convolution1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.normalization1 = nn.BatchNorm2d(mid_channels)
        self.activation = nn.ReLU(inplace=True)
        self.convolution2 = nn.Conv2d(
            mid_channels, out_channels, kernel_size=3, padding=1, bias=False
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


class ResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        expanded_channels = out_channels * self.expansion
        self.convolution1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.normalization1 = nn.BatchNorm2d(out_channels)
        self.convolution2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.normalization2 = nn.BatchNorm2d(out_channels)
        self.convolution3 = nn.Conv2d(
            out_channels, expanded_channels, kernel_size=1, bias=False
        )
        self.normalization3 = nn.BatchNorm2d(expanded_channels)
        self.activation = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != expanded_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    expanded_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(expanded_channels),
            )

    def forward(self, inputs):
        residual = self.shortcut(inputs)
        outputs = self.activation(self.normalization1(self.convolution1(inputs)))
        outputs = self.activation(self.normalization2(self.convolution2(outputs)))
        outputs = self.normalization3(self.convolution3(outputs))
        return self.activation(outputs + residual)


class ResNetResidualAdd(nn.Module):
    """Standalone residual add with the same tensor shape on both branches."""

    def forward(self, inputs):
        return inputs + inputs


class ResNet18(nn.Module):
    def __init__(self, num_classes, cifar_stem=False):
        super().__init__()
        stem_conv = (
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if cifar_stem
            else nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        )
        self.stem = nn.Sequential(
            stem_conv,
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


DEFAULT_PRUNED_CONFIG = {
    "stem_out": 64,
    "blocks": [
        # stage 1 (layer1.0, layer1.1)
        {"in_channels": 64, "mid_channels": 64, "out_channels": 64, "stride": 1},
        {"in_channels": 64, "mid_channels": 58, "out_channels": 64, "stride": 1},
        # stage 2 (layer2.0, layer2.1)
        {"in_channels": 64, "mid_channels": 128, "out_channels": 127, "stride": 2},
        {"in_channels": 127, "mid_channels": 128, "out_channels": 127, "stride": 1},
        # stage 3 (layer3.0, layer3.1)
        {"in_channels": 127, "mid_channels": 253, "out_channels": 249, "stride": 2},
        {"in_channels": 249, "mid_channels": 192, "out_channels": 236, "stride": 1},
        # stage 4 (layer4.0, layer4.1)
        {"in_channels": 236, "mid_channels": 381, "out_channels": 222, "stride": 2},
        {"in_channels": 222, "mid_channels": 16, "out_channels": 297, "stride": 1},
    ],
    "classifier_in": 297,
}


class PrunedCifarResNet18(nn.Module):
    """Pruned ResNet-18 architecture reconstructed from JouleNAS channel pruning."""

    def __init__(self, num_classes=10, cifar_stem=True, config=None):
        super().__init__()
        cfg = config or DEFAULT_PRUNED_CONFIG
        stem_out = cfg["stem_out"]
        stem_conv = (
            nn.Conv2d(3, stem_out, kernel_size=3, stride=1, padding=1, bias=False)
            if cifar_stem
            else nn.Conv2d(3, stem_out, kernel_size=7, stride=2, padding=3, bias=False)
        )
        self.stem = nn.Sequential(
            stem_conv,
            nn.BatchNorm2d(stem_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        blocks = cfg["blocks"]
        self.layer1 = nn.Sequential(
            ResNetBasicBlock(**blocks[0]),
            ResNetBasicBlock(**blocks[1]),
        )
        self.layer2 = nn.Sequential(
            ResNetBasicBlock(**blocks[2]),
            ResNetBasicBlock(**blocks[3]),
        )
        self.layer3 = nn.Sequential(
            ResNetBasicBlock(**blocks[4]),
            ResNetBasicBlock(**blocks[5]),
        )
        self.layer4 = nn.Sequential(
            ResNetBasicBlock(**blocks[6]),
            ResNetBasicBlock(**blocks[7]),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(cfg["classifier_in"], num_classes)

    @classmethod
    def from_pruning_log(cls, path, num_classes=10, cifar_stem=True):
        """Build a PrunedCifarResNet18 directly from a JouleNAS pruning log JSON."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        stem_out = data["stem.conv"]["active_output"]
        blocks = [
            {
                "in_channels": data["layer1.0.conv1"]["active_input"],
                "mid_channels": data["layer1.0.conv1"]["active_output"],
                "out_channels": data["layer1.0.conv2"]["active_output"],
                "stride": 1,
            },
            {
                "in_channels": data["layer1.1.conv1"]["active_input"],
                "mid_channels": data["layer1.1.conv1"]["active_output"],
                "out_channels": data["layer1.1.conv2"]["active_output"],
                "stride": 1,
            },
            {
                "in_channels": data["layer2.0.conv1"]["active_input"],
                "mid_channels": data["layer2.0.conv1"]["active_output"],
                "out_channels": data["layer2.0.conv2"]["active_output"],
                "stride": 2,
            },
            {
                "in_channels": data["layer2.1.conv1"]["active_input"],
                "mid_channels": data["layer2.1.conv1"]["active_output"],
                "out_channels": data["layer2.1.conv2"]["active_output"],
                "stride": 1,
            },
            {
                "in_channels": data["layer3.0.conv1"]["active_input"],
                "mid_channels": data["layer3.0.conv1"]["active_output"],
                "out_channels": data["layer3.0.conv2"]["active_output"],
                "stride": 2,
            },
            {
                "in_channels": data["layer3.1.conv1"]["active_input"],
                "mid_channels": data["layer3.1.conv1"]["active_output"],
                "out_channels": data["layer3.1.conv2"]["active_output"],
                "stride": 1,
            },
            {
                "in_channels": data["layer4.0.conv1"]["active_input"],
                "mid_channels": data["layer4.0.conv1"]["active_output"],
                "out_channels": data["layer4.0.conv2"]["active_output"],
                "stride": 2,
            },
            {
                "in_channels": data["layer4.1.conv1"]["active_input"],
                "mid_channels": data["layer4.1.conv1"]["active_output"],
                "out_channels": data["layer4.1.conv2"]["active_output"],
                "stride": 1,
            },
        ]
        classifier_in = data["classifier"]["active_input"]
        config = {
            "stem_out": stem_out,
            "blocks": blocks,
            "classifier_in": classifier_in,
        }
        return cls(num_classes=num_classes, cifar_stem=cifar_stem, config=config)

    def forward(self, inputs):
        outputs = self.stem(inputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.pool(outputs)
        return self.classifier(torch.flatten(outputs, 1))


PrunedResNet18 = PrunedCifarResNet18


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, blocks=3)
        self.layer2 = self._make_layer(256, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(512, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(1024, 512, blocks=3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, num_classes)

    @staticmethod
    def _make_layer(in_channels, out_channels, blocks, stride=1):
        layers = [ResNetBottleneck(in_channels, out_channels, stride)]
        expanded_channels = out_channels * ResNetBottleneck.expansion
        layers.extend(
            ResNetBottleneck(expanded_channels, out_channels)
            for _ in range(blocks - 1)
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