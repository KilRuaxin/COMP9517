import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights  # ✅ 新增 weights 枚举

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, include_top=True, pretrained=False):
        super(MobileNetV2, self).__init__()
        self.include_top = include_top

        if pretrained:
            weights = MobileNet_V2_Weights.DEFAULT  # ✅ 加载官方推荐权重
        else:
            weights = None

        self.base_model = mobilenet_v2(weights=weights)

        if not include_top:
            self.base_model.classifier = nn.Identity()
        else:
            in_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features, num_classes)
            )

    def forward(self, x):
        return self.base_model(x)


def mobilenet_v2_custom(num_classes=1000, include_top=True, pretrained=False):
    return MobileNetV2(num_classes=num_classes, include_top=include_top, pretrained=pretrained)
#