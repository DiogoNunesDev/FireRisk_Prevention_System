import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(DeepLabV3Plus, self).__init__()
        self.model = deeplabv3_resnet101(weights="DEFAULT" if pretrained else None)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)['out'] 