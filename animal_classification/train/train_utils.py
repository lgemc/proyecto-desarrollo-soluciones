import torch
import numpy as np
import random
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import json
from collections import defaultdict


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score: float) -> bool:
        if self.mode == 'min':
            score = -val_score
        else:
            score = val_score
            
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            
        return self.early_stop


class MetricsTracker:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.best_metrics = {}
        
    def update(self, metrics_dict: Dict[str, float]):
        for key, value in metrics_dict.items():
            self.metrics[key].append(value)
            
    def update_best(self, metric_name: str, value: float, mode: str = 'max'):
        if metric_name not in self.best_metrics:
            self.best_metrics[metric_name] = value
        else:
            if mode == 'max' and value > self.best_metrics[metric_name]:
                self.best_metrics[metric_name] = value
            elif mode == 'min' and value < self.best_metrics[metric_name]:
                self.best_metrics[metric_name] = value
                
    def get_last(self, metric_name: str) -> Optional[float]:
        if metric_name in self.metrics and len(self.metrics[metric_name]) > 0:
            return self.metrics[metric_name][-1]
        return None
    
    def get_average(self, metric_name: str, last_n: int = None) -> Optional[float]:
        if metric_name in self.metrics:
            values = self.metrics[metric_name]
            if last_n:
                values = values[-last_n:]
            if len(values) > 0:
                return np.mean(values)
        return None
    
    def save(self, filepath: Path):
        with open(filepath, 'w') as f:
            json.dump({
                'metrics': dict(self.metrics),
                'best_metrics': self.best_metrics
            }, f, indent=2)
    
    def load(self, filepath: Path):
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.metrics = defaultdict(list, data['metrics'])
            self.best_metrics = data['best_metrics']


def create_data_loaders(
    dataset,
    config,
    augmentation: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    total_size = len(dataset)
    train_size = int(config.train_split * total_size)
    val_size = int(config.val_split * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    if augmentation and config.augmentation:
        aug_transforms = []
        
        if 'random_horizontal_flip' in config.augmentation_params:
            aug_transforms.append(
                transforms.RandomHorizontalFlip(p=config.augmentation_params['random_horizontal_flip'])
            )
        
        if 'random_rotation' in config.augmentation_params:
            aug_transforms.append(
                transforms.RandomRotation(degrees=config.augmentation_params['random_rotation'])
            )
        
        if 'color_jitter' in config.augmentation_params:
            params = config.augmentation_params['color_jitter']
            aug_transforms.append(
                transforms.ColorJitter(**params)
            )
        
        if 'random_resized_crop' in config.augmentation_params:
            params = config.augmentation_params['random_resized_crop']
            aug_transforms.append(
                transforms.RandomResizedCrop(**params)
            )
        
        aug_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_transform = transforms.Compose(aug_transforms)
        
        from animal_classification.datasets import AnimalImageDataset
        train_dataset = AnimalImageDataset(
            data_dir=config.data_dir,
            transform=train_transform
        )
        
        train_indices = train_dataset.dataset.indices if hasattr(train_dataset, 'dataset') else range(train_size)
        train_dataset = Subset(train_dataset, train_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    return train_loader, val_loader, test_loader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct / total * 100