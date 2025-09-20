import torch
import torch.nn as nn
import timm

class HRNetSegmentation(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(HRNetSegmentation, self).__init__()
        
        # Backbone HRNet com saída de features
        self.backbone = timm.create_model(
            "hrnet_w18_small_v2.ms_in1k",
            features_only=True,
            pretrained=pretrained
        )

        # Obtém a dimensão da última camada de features
        last_channel_dim = self.backbone.feature_info[-1]['num_chs']

        # Decoder simples (1x1 conv para mapeamento para classes)
        self.classifier = nn.Sequential(
            nn.Conv2d(last_channel_dim, num_classes, kernel_size=1)
        )

    def forward(self, x):
        features = self.backbone(x)[-1]  
        out = self.classifier(features)
        # Upsample para o tamanho original 
        out = nn.functional.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return out
