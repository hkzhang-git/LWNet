import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, dropout_keep_prob=0):
        self.inplanes = 32
        super(ResNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(8, 3, 3), stride=1, padding=0, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 3, 3), stride=2),
            self._make_layer(block, 32, layers[0]),
            self._make_layer(block, 64, layers[1], stride=2),
            self._make_layer(block, 128, layers[2], stride=2),
            self._make_layer(block, 256, layers[3], stride=2),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            # nn.Dropout3d(dropout_keep_prob)
        )
        self.fc = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze(2).squeeze(2).squeeze(2)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)


def resnet10(**kwargs):
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def resnet14_a(**kwargs):
    model = ResNet(BasicBlock, [1, 2, 2, 1], **kwargs)
    return model


def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet38(**kwargs):
    model = ResNet(BasicBlock, [3, 5, 7, 3], **kwargs)
    return model


def resnet14_b(**kwargs):
    model = ResNet(Bottleneck, [1, 1, 1, 1], **kwargs)
    return model


def resnet20(**kwargs):
    model = ResNet(Bottleneck, [1, 2, 2, 1], **kwargs)
    return model


def resnet26(**kwargs):
    model = ResNet(Bottleneck, [2, 2, 2, 2], **kwargs)
    return model


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet56(**kwargs):
    model = ResNet(Bottleneck, [3, 5, 7, 3], **kwargs)
    return model



dict={'resnet10':resnet10, 'resnet14_a':resnet14_a, 'resnet14_b':resnet14_b, 'resnet18':resnet18, 'resnet20':resnet20, 'resnet26': resnet26, 'resnet34':resnet34, 'resnet38':resnet38, 'resnet50':resnet50, 'resnet56':resnet56}
