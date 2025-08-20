#!/usr/bin/env python3
"""Unit tests for the Trainer class."""

import sys
import unittest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from animal_classification.train.trainer import Trainer
from animal_classification.train.train_config import TrainConfig
from animal_classification.models import ResNetConfig


class TestTrainer(unittest.TestCase):
    """Test cases for the Trainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a minimal config for testing
        self.config = TrainConfig(
            data_dir='test_data',
            output_dir='test_outputs',
            num_epochs=2,
            batch_size=8,
            device='cpu',  # Use CPU for testing
            save_frequency=1,
            early_stopping=False,
            use_amp=False  # Disable AMP for testing
        )
        
        self.model_config = ResNetConfig(
            resnet_variant='resnet18',
            pretrained=False  # Don't download weights during testing
        )
        
        # Mock the dataset and data loaders
        self.mock_dataset_patcher = patch('animal_classification.train.trainer.AnimalImageDataset')
        self.mock_data_loader_patcher = patch('animal_classification.train.trainer.create_data_loaders')
        
    def tearDown(self):
        """Clean up after tests."""
        # Clean up test directories if they exist
        import shutil
        if Path('test_outputs').exists():
            shutil.rmtree('test_outputs')
    
    def _create_mock_data_loader(self, num_batches=5, batch_size=8):
        """Create a mock data loader for testing."""
        mock_loader = []
        for _ in range(num_batches):
            # Create fake batch data
            images = torch.randn(batch_size, 3, 224, 224)
            labels = torch.randint(0, 4, (batch_size,))
            mock_loader.append((images, labels))
        return mock_loader
    
    @patch('animal_classification.train.trainer.AnimalClassifier')
    def test_trainer_initialization(self, mock_classifier):
        """Test that the Trainer initializes correctly."""
        with self.mock_dataset_patcher as mock_dataset_class, \
             self.mock_data_loader_patcher as mock_create_loaders:
            
            # Setup mocks
            mock_dataset = MagicMock()
            mock_dataset.get_num_classes.return_value = 4
            mock_dataset.get_class_names.return_value = ['cat', 'dog', 'bird', 'fish']
            mock_dataset.__len__.return_value = 100
            mock_dataset_class.return_value = mock_dataset
            
            mock_train_loader = self._create_mock_data_loader(5, 8)
            mock_val_loader = self._create_mock_data_loader(2, 8)
            mock_test_loader = self._create_mock_data_loader(2, 8)
            mock_create_loaders.return_value = (mock_train_loader, mock_val_loader, mock_test_loader)
            
            mock_model = MagicMock()
            mock_classifier.from_dataset.return_value = mock_model
            
            # Create trainer
            trainer = Trainer(self.config, self.model_config)
            
            # Assertions
            self.assertIsNotNone(trainer)
            self.assertEqual(trainer.config, self.config)
            self.assertEqual(trainer.model_config, self.model_config)
            self.assertEqual(trainer.device.type, 'cpu')
            self.assertIsNotNone(trainer.optimizer)
            self.assertIsNotNone(trainer.criterion)
            self.assertIsNone(trainer.early_stopping)  # Disabled in config
            self.assertEqual(trainer.current_epoch, 0)
            self.assertEqual(trainer.best_val_acc, 0)
            self.assertEqual(trainer.best_val_loss, float('inf'))
    
    def test_create_optimizer(self):
        """Test optimizer creation with different configurations."""
        with self.mock_dataset_patcher as mock_dataset_class, \
             self.mock_data_loader_patcher as mock_create_loaders:
            
            # Setup mocks
            mock_dataset = MagicMock()
            mock_dataset.get_num_classes.return_value = 4
            mock_dataset_class.return_value = mock_dataset
            mock_create_loaders.return_value = ([], [], [])
            
            # Test Adam optimizer
            self.config.optimizer = 'adam'
            with patch('animal_classification.train.trainer.AnimalClassifier'):
                trainer = Trainer(self.config, self.model_config)
                self.assertIsInstance(trainer.optimizer, torch.optim.Adam)
            
            # Test AdamW optimizer
            self.config.optimizer = 'adamw'
            with patch('animal_classification.train.trainer.AnimalClassifier'):
                trainer = Trainer(self.config, self.model_config)
                self.assertIsInstance(trainer.optimizer, torch.optim.AdamW)
            
            # Test SGD optimizer
            self.config.optimizer = 'sgd'
            with patch('animal_classification.train.trainer.AnimalClassifier'):
                trainer = Trainer(self.config, self.model_config)
                self.assertIsInstance(trainer.optimizer, torch.optim.SGD)
            
            # Test unknown optimizer
            self.config.optimizer = 'unknown'
            with patch('animal_classification.train.trainer.AnimalClassifier'):
                with self.assertRaises(ValueError):
                    trainer = Trainer(self.config, self.model_config)
    
    def test_create_scheduler(self):
        """Test scheduler creation with different configurations."""
        with self.mock_dataset_patcher as mock_dataset_class, \
             self.mock_data_loader_patcher as mock_create_loaders:
            
            # Setup mocks
            mock_dataset = MagicMock()
            mock_dataset.get_num_classes.return_value = 4
            mock_dataset_class.return_value = mock_dataset
            mock_create_loaders.return_value = ([], [], [])
            
            # Test cosine scheduler
            self.config.scheduler = 'cosine'
            self.config.optimizer = 'adam'  # Reset to valid optimizer
            with patch('animal_classification.train.trainer.AnimalClassifier'):
                trainer = Trainer(self.config, self.model_config)
                self.assertIsInstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
            
            # Test step scheduler
            self.config.scheduler = 'step'
            self.config.scheduler_params = {'step_size': 10, 'gamma': 0.1}
            with patch('animal_classification.train.trainer.AnimalClassifier'):
                trainer = Trainer(self.config, self.model_config)
                self.assertIsInstance(trainer.scheduler, torch.optim.lr_scheduler.StepLR)
            
            # Test no scheduler
            self.config.scheduler = None
            with patch('animal_classification.train.trainer.AnimalClassifier'):
                trainer = Trainer(self.config, self.model_config)
                self.assertIsNone(trainer.scheduler)
    
    @patch('animal_classification.train.trainer.AnimalClassifier')
    def test_train_epoch(self, mock_classifier):
        """Test training for one epoch."""
        with self.mock_dataset_patcher as mock_dataset_class, \
             self.mock_data_loader_patcher as mock_create_loaders:
            
            # Setup mocks
            mock_dataset = MagicMock()
            mock_dataset.get_num_classes.return_value = 4
            mock_dataset.get_class_names.return_value = ['cat', 'dog', 'bird', 'fish']
            mock_dataset.__len__.return_value = 100
            mock_dataset_class.return_value = mock_dataset
            
            mock_train_loader = self._create_mock_data_loader(5, 8)
            mock_val_loader = self._create_mock_data_loader(2, 8)
            mock_test_loader = self._create_mock_data_loader(2, 8)
            mock_create_loaders.return_value = (mock_train_loader, mock_val_loader, mock_test_loader)
            
            # Create a simple mock model
            mock_model = MagicMock()
            mock_model.train = MagicMock()
            mock_model.eval = MagicMock()
            mock_model.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(2, 2))])
            mock_model.forward = MagicMock(return_value=torch.randn(8, 4))  # batch_size x num_classes
            mock_model.to = MagicMock(return_value=mock_model)
            mock_classifier.from_dataset.return_value = mock_model
            
            # Create trainer
            trainer = Trainer(self.config, self.model_config)
            
            # Run one training epoch
            with patch('animal_classification.train.trainer.tqdm', side_effect=lambda x, **kwargs: x):
                metrics = trainer.train_epoch()
            
            # Assertions
            self.assertIn('train_loss', metrics)
            self.assertIn('train_acc', metrics)
            self.assertIsInstance(metrics['train_loss'], float)
            self.assertIsInstance(metrics['train_acc'], float)
            mock_model.train.assert_called()
    
    @patch('animal_classification.train.trainer.AnimalClassifier')
    def test_validate(self, mock_classifier):
        """Test validation."""
        with self.mock_dataset_patcher as mock_dataset_class, \
             self.mock_data_loader_patcher as mock_create_loaders:
            
            # Setup mocks
            mock_dataset = MagicMock()
            mock_dataset.get_num_classes.return_value = 4
            mock_dataset_class.return_value = mock_dataset
            
            mock_train_loader = self._create_mock_data_loader(5, 8)
            mock_val_loader = self._create_mock_data_loader(2, 8)
            mock_test_loader = self._create_mock_data_loader(2, 8)
            mock_create_loaders.return_value = (mock_train_loader, mock_val_loader, mock_test_loader)
            
            # Create a simple mock model
            mock_model = MagicMock()
            mock_model.eval = MagicMock()
            mock_model.forward = MagicMock(return_value=torch.randn(8, 4))
            mock_model.to = MagicMock(return_value=mock_model)
            mock_classifier.from_dataset.return_value = mock_model
            
            # Create trainer
            trainer = Trainer(self.config, self.model_config)
            
            # Run validation
            with patch('animal_classification.train.trainer.tqdm', side_effect=lambda x, **kwargs: x):
                metrics = trainer.validate()
            
            # Assertions
            self.assertIn('val_loss', metrics)
            self.assertIn('val_acc', metrics)
            self.assertIsInstance(metrics['val_loss'], float)
            self.assertIsInstance(metrics['val_acc'], float)
            mock_model.eval.assert_called()
    
    @patch('animal_classification.train.trainer.AnimalClassifier')
    @patch('torch.save')
    def test_save_checkpoint(self, mock_save, mock_classifier):
        """Test checkpoint saving."""
        with self.mock_dataset_patcher as mock_dataset_class, \
             self.mock_data_loader_patcher as mock_create_loaders:
            
            # Setup mocks
            mock_dataset = MagicMock()
            mock_dataset.get_num_classes.return_value = 4
            mock_dataset_class.return_value = mock_dataset
            mock_create_loaders.return_value = ([], [], [])
            
            mock_model = MagicMock()
            mock_model.state_dict.return_value = {'layer': 'weights'}
            mock_model.to = MagicMock(return_value=mock_model)
            mock_classifier.from_dataset.return_value = mock_model
            
            # Create trainer
            trainer = Trainer(self.config, self.model_config)
            trainer.current_epoch = 5
            trainer.best_val_acc = 85.0
            
            # Save checkpoint
            trainer.save_checkpoint(is_best=True)
            
            # Assertions
            self.assertEqual(mock_save.call_count, 2)  # Regular and best checkpoint
            saved_checkpoint = mock_save.call_args_list[0][0][0]
            self.assertIn('epoch', saved_checkpoint)
            self.assertIn('model_state_dict', saved_checkpoint)
            self.assertIn('optimizer_state_dict', saved_checkpoint)
            self.assertEqual(saved_checkpoint['epoch'], 5)
            self.assertEqual(saved_checkpoint['best_val_acc'], 85.0)
    
    @patch('animal_classification.train.trainer.AnimalClassifier')
    @patch('torch.load')
    def test_load_checkpoint(self, mock_load, mock_classifier):
        """Test checkpoint loading."""
        with self.mock_dataset_patcher as mock_dataset_class, \
             self.mock_data_loader_patcher as mock_create_loaders:
            
            # Setup mocks
            mock_dataset = MagicMock()
            mock_dataset.get_num_classes.return_value = 4
            mock_dataset_class.return_value = mock_dataset
            mock_create_loaders.return_value = ([], [], [])
            
            mock_model = MagicMock()
            mock_model.load_state_dict = MagicMock()
            mock_model.to = MagicMock(return_value=mock_model)
            mock_classifier.from_dataset.return_value = mock_model
            
            # Mock checkpoint data
            checkpoint_data = {
                'epoch': 10,
                'model_state_dict': {'layer': 'weights'},
                'optimizer_state_dict': {'lr': 0.001},
                'scheduler_state_dict': {'last_epoch': 10},
                'best_val_acc': 90.0,
                'best_val_loss': 0.25
            }
            mock_load.return_value = checkpoint_data
            
            # Create trainer
            trainer = Trainer(self.config, self.model_config)
            
            # Load checkpoint
            trainer.load_checkpoint(Path('test_checkpoint.pth'))
            
            # Assertions
            mock_load.assert_called_once()
            mock_model.load_state_dict.assert_called_with({'layer': 'weights'})
            self.assertEqual(trainer.current_epoch, 10)
            self.assertEqual(trainer.best_val_acc, 90.0)
            self.assertEqual(trainer.best_val_loss, 0.25)
    
    @patch('animal_classification.train.trainer.AnimalClassifier')
    @patch('torch.save')
    def test_train_method(self, mock_save, mock_classifier):
        """Test the main training loop."""
        with self.mock_dataset_patcher as mock_dataset_class, \
             self.mock_data_loader_patcher as mock_create_loaders:
            
            # Setup mocks
            mock_dataset = MagicMock()
            mock_dataset.get_num_classes.return_value = 4
            mock_dataset.get_class_names.return_value = ['cat', 'dog', 'bird', 'fish']
            mock_dataset.__len__.return_value = 100
            mock_dataset_class.return_value = mock_dataset
            
            # Create smaller loaders for faster testing
            mock_train_loader = self._create_mock_data_loader(2, 4)
            mock_val_loader = self._create_mock_data_loader(1, 4)
            mock_test_loader = self._create_mock_data_loader(1, 4)
            mock_create_loaders.return_value = (mock_train_loader, mock_val_loader, mock_test_loader)
            
            # Create a simple mock model
            mock_model = MagicMock()
            mock_model.train = MagicMock()
            mock_model.eval = MagicMock()
            mock_model.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(2, 2))])
            mock_model.forward = MagicMock(return_value=torch.randn(4, 4))
            mock_model.state_dict.return_value = {'layer': 'weights'}
            mock_model.to = MagicMock(return_value=mock_model)
            mock_classifier.from_dataset.return_value = mock_model
            
            # Update config for faster testing
            self.config.num_epochs = 1
            self.config.save_frequency = 1
            
            # Create trainer
            trainer = Trainer(self.config, self.model_config)
            
            # Run training
            with patch('animal_classification.train.trainer.tqdm', side_effect=lambda x, **kwargs: x), \
                 patch('builtins.print'):  # Suppress print statements
                metrics_tracker = trainer.train()
            
            # Assertions
            self.assertIsNotNone(metrics_tracker)
            self.assertEqual(trainer.current_epoch, 0)  # Should be 0 after 1 epoch (0-indexed)
            mock_save.assert_called()  # Should save checkpoints
    
    @patch('animal_classification.train.trainer.AnimalClassifier')
    def test_early_stopping(self, mock_classifier):
        """Test early stopping functionality."""
        # Enable early stopping
        self.config.early_stopping = True
        self.config.early_stopping_patience = 2
        self.config.num_epochs = 10  # Set high to test early stopping
        
        with self.mock_dataset_patcher as mock_dataset_class, \
             self.mock_data_loader_patcher as mock_create_loaders, \
             patch('animal_classification.train.trainer.EarlyStopping') as mock_early_stopping_class:
            
            # Setup mocks
            mock_dataset = MagicMock()
            mock_dataset.get_num_classes.return_value = 4
            mock_dataset.get_class_names.return_value = ['cat', 'dog', 'bird', 'fish']
            mock_dataset.__len__.return_value = 100
            mock_dataset_class.return_value = mock_dataset
            
            mock_train_loader = self._create_mock_data_loader(1, 4)
            mock_val_loader = self._create_mock_data_loader(1, 4)
            mock_test_loader = self._create_mock_data_loader(1, 4)
            mock_create_loaders.return_value = (mock_train_loader, mock_val_loader, mock_test_loader)
            
            # Mock model
            mock_model = MagicMock()
            mock_model.train = MagicMock()
            mock_model.eval = MagicMock()
            mock_model.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(2, 2))])
            mock_model.forward = MagicMock(return_value=torch.randn(4, 4))
            mock_model.state_dict.return_value = {'layer': 'weights'}
            mock_model.to = MagicMock(return_value=mock_model)
            mock_classifier.from_dataset.return_value = mock_model
            
            # Mock early stopping to trigger after 3 epochs
            mock_early_stopping = MagicMock()
            mock_early_stopping.side_effect = [False, False, True]  # Stop on third call
            mock_early_stopping_class.return_value = mock_early_stopping
            
            # Create trainer
            trainer = Trainer(self.config, self.model_config)
            
            # Run training
            with patch('animal_classification.train.trainer.tqdm', side_effect=lambda x, **kwargs: x), \
                 patch('builtins.print'), \
                 patch('torch.save'):
                metrics_tracker = trainer.train()
            
            # Assertions - should stop early
            self.assertEqual(trainer.current_epoch, 2)  # Should stop after 3 epochs (0-indexed)
            self.assertEqual(mock_early_stopping.call_count, 3)


if __name__ == "__main__":
    unittest.main()