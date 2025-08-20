import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Literal
from dataclasses import dataclass


@dataclass
class ResNetConfig:
    resnet_variant: Literal['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'] = 'resnet50'
    num_classes: int = 10
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout_rate: float = 0.2
    hidden_size: Optional[int] = 512


class AnimalClassifier(nn.Module):
    def __init__(self, config: ResNetConfig):
        super(AnimalClassifier, self).__init__()
        self.config = config
        
        resnet_models = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152
        }
        
        if config.resnet_variant not in resnet_models:
            raise ValueError(f"Invalid ResNet variant: {config.resnet_variant}")
        
        self.backbone = resnet_models[config.resnet_variant](
            weights='IMAGENET1K_V1' if config.pretrained else None
        )
        
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        num_features = self.backbone.fc.in_features
        
        if config.hidden_size:
            self.backbone.fc = nn.Sequential(
                nn.Linear(num_features, config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.hidden_size, config.num_classes)
            )
        else:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(config.dropout_rate),
                nn.Linear(num_features, config.num_classes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_feature_extractor(self) -> nn.Module:
        feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        return feature_extractor
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feature_extractor = self.get_feature_extractor()
        features = feature_extractor(x)
        features = torch.flatten(features, 1)
        return features
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities
    
    @classmethod
    def from_dataset(cls, dataset, config: Optional[ResNetConfig] = None):
        if config is None:
            config = ResNetConfig()
        
        config.num_classes = dataset.get_num_classes()
        
        return cls(config)
    
    def unfreeze_backbone(self, unfreeze_last_n_layers: Optional[int] = None):
        if unfreeze_last_n_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            layers = list(self.backbone.children())
            for layer in layers[-unfreeze_last_n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
    
    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False