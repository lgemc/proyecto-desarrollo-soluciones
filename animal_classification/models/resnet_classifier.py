from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

class ResnetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_feats = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_feats, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, num_classes),
        )
    def forward(self, x):
        return self.backbone(x)  # returns logits