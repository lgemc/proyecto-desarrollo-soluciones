import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

from animal_classification.datasets import AnimalImageDataset
from animal_classification.models import AnimalClassifier, ResNetConfig
from .train_config import TrainConfig
from .train_utils import (
    EarlyStopping, MetricsTracker, create_data_loaders, 
    set_seed, mixup_data, cutmix_data, calculate_accuracy
)


class Trainer:
    def __init__(
        self,
        config: TrainConfig,
        model_config: Optional[ResNetConfig] = None
    ):
        self.config = config
        self.model_config = model_config or ResNetConfig()
        
        set_seed(config.seed)
        
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        self.dataset = AnimalImageDataset(data_dir=config.data_dir)
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            self.dataset, config
        )
        
        self.model = AnimalClassifier.from_dataset(self.dataset, self.model_config)
        self.model = self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        self.scaler = GradScaler() if config.use_amp else None
        
        self.metrics_tracker = MetricsTracker()
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta
        ) if config.early_stopping else None
        
        self.current_epoch = 0
        self.best_val_acc = 0
        self.best_val_loss = float('inf')
        
    def _create_optimizer(self) -> optim.Optimizer:
        if self.config.optimizer == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[Any]:
        if self.config.scheduler is None:
            return None
            
        if self.config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                **self.config.scheduler_params
            )
        elif self.config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                **self.config.scheduler_params
            )
        elif self.config.scheduler == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                **self.config.scheduler_params
            )
        elif self.config.scheduler == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                **self.config.scheduler_params
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
    
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        running_acc = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}/{self.config.num_epochs}')
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            if self.config.mixup and np.random.rand() > 0.5:
                inputs, targets_a, targets_b, lam = mixup_data(
                    inputs, targets, self.config.mixup_alpha
                )
                
            elif self.config.cutmix and np.random.rand() > 0.5:
                inputs, targets_a, targets_b, lam = cutmix_data(
                    inputs, targets, self.config.cutmix_alpha
                )
            else:
                targets_a = targets_b = targets
                lam = 1
            
            self.optimizer.zero_grad()
            
            if self.config.use_amp and self.scaler:
                with autocast():
                    outputs = self.model(inputs)
                    if lam == 1:
                        loss = self.criterion(outputs, targets)
                    else:
                        loss = lam * self.criterion(outputs, targets_a) + \
                               (1 - lam) * self.criterion(outputs, targets_b)
                
                self.scaler.scale(loss).backward()
                
                if self.config.gradient_clip_val:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_val
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                if lam == 1:
                    loss = self.criterion(outputs, targets)
                else:
                    loss = lam * self.criterion(outputs, targets_a) + \
                           (1 - lam) * self.criterion(outputs, targets_b)
                
                loss.backward()
                
                if self.config.gradient_clip_val:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_val
                    )
                
                self.optimizer.step()
            
            running_loss += loss.item()
            
            if lam == 1:
                acc = calculate_accuracy(outputs, targets)
            else:
                acc = lam * calculate_accuracy(outputs, targets_a) + \
                      (1 - lam) * calculate_accuracy(outputs, targets_b)
            running_acc += acc
            
            if batch_idx % self.config.log_interval == 0:
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': f'{acc:.2f}%',
                    'lr': self.optimizer.param_groups[0]['lr']
                })
        
        epoch_loss = running_loss / num_batches
        epoch_acc = running_acc / num_batches
        
        return {'train_loss': epoch_loss, 'train_acc': epoch_acc}
    
    def validate(self, loader=None) -> Dict[str, float]:
        if loader is None:
            loader = self.val_loader
            
        self.model.eval()
        running_loss = 0.0
        running_acc = 0.0
        num_batches = len(loader)
        
        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc='Validation'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                running_acc += calculate_accuracy(outputs, targets)
        
        val_loss = running_loss / num_batches
        val_acc = running_acc / num_batches
        
        return {'val_loss': val_loss, 'val_acc': val_acc}
    
    def test(self) -> Dict[str, float]:
        test_metrics = self.validate(self.test_loader)
        return {
            'test_loss': test_metrics['val_loss'],
            'test_acc': test_metrics['val_acc']
        }
    
    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__,
            'model_config': self.model_config.__dict__
        }
        
        checkpoint_path = self.config.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.config.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            
        if not self.config.save_best_only:
            latest_path = self.config.checkpoint_dir / 'latest_model.pth'
            torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_loss = checkpoint['best_val_loss']
    
    def train(self):
        print(f"Starting training on {self.device}")
        print(f"Model: {self.model_config.resnet_variant}")
        print(f"Dataset: {len(self.dataset)} images, {self.dataset.get_num_classes()} classes")
        print(f"Classes: {', '.join(self.dataset.get_class_names())}")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            if self.config.progressive_unfreezing and epoch == self.config.unfreeze_after_epoch:
                print("Unfreezing backbone layers...")
                self.model.unfreeze_backbone()
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            all_metrics = {**train_metrics, **val_metrics}
            self.metrics_tracker.update(all_metrics)
            
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            print(f"Train Loss: {train_metrics['train_loss']:.4f}, Train Acc: {train_metrics['train_acc']:.2f}%")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_acc']:.2f}%")
            
            is_best = val_metrics['val_acc'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['val_acc']
                self.best_val_loss = val_metrics['val_loss']
                print(f"New best model! Val Acc: {self.best_val_acc:.2f}%")
            
            if epoch % self.config.save_frequency == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
            
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            if self.early_stopping:
                if self.early_stopping(val_metrics['val_loss']):
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        print("\nTraining completed!")
        print(f"Best Val Acc: {self.best_val_acc:.2f}%")
        
        print("\nEvaluating on test set...")
        test_metrics = self.test()
        print(f"Test Loss: {test_metrics['test_loss']:.4f}, Test Acc: {test_metrics['test_acc']:.2f}%")
        
        self.metrics_tracker.update_best('test_acc', test_metrics['test_acc'])
        self.metrics_tracker.save(self.config.log_dir / 'metrics.json')
        
        return self.metrics_tracker