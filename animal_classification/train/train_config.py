from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal
from pathlib import Path


@dataclass
class TrainConfig:
    data_dir: str = 'data'
    output_dir: str = 'outputs'
    model_name: str = 'animal_classifier'
    
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    optimizer: Literal['adam', 'adamw', 'sgd'] = 'adamw'
    scheduler: Optional[Literal['cosine', 'step', 'exponential', 'reduce_on_plateau']] = 'cosine'
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        'T_max': 50,
        'eta_min': 1e-6
    })
    
    warmup_epochs: int = 5
    gradient_clip_val: Optional[float] = 1.0
    
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    save_best_only: bool = True
    save_frequency: int = 5
    
    num_workers: int = 4
    pin_memory: bool = True
    
    use_amp: bool = True
    
    augmentation: bool = True
    augmentation_params: Dict[str, Any] = field(default_factory=lambda: {
        'random_horizontal_flip': 0.5,
        'random_rotation': 15,
        'color_jitter': {'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1},
        'random_resized_crop': {'size': 224, 'scale': (0.8, 1.0)}
    })
    
    label_smoothing: float = 0.1
    
    progressive_unfreezing: bool = False
    unfreeze_after_epoch: int = 10
    
    device: str = 'cuda'
    seed: int = 42
    
    log_interval: int = 10
    val_check_interval: float = 1.0
    
    mixup: bool = False
    mixup_alpha: float = 0.2
    
    cutmix: bool = False
    cutmix_alpha: float = 1.0
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = self.output_dir / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)