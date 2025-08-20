#!/usr/bin/env python3
"""Test script for animal classification datasets."""

import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from animal_classification.datasets import AnimalMetadataDataset, AnimalImageDataset
import torch
from torch.utils.data import DataLoader


class TestAnimalDatasets(unittest.TestCase):
    """Test cases for animal classification datasets."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_path = '../../data'
    
    def test_metadata_dataset(self):
        """Test the metadata dataset."""
        dataset = AnimalMetadataDataset(self.data_path)
        
        # Test dataset properties
        self.assertIsInstance(dataset, AnimalMetadataDataset)
        self.assertGreaterEqual(len(dataset), 0)
        self.assertIsInstance(dataset.get_num_classes(), int)
        self.assertIsInstance(dataset.get_class_names(), list)
        print(f"Number of classes: {dataset.get_num_classes()}")
        self.assertEqual(len(dataset.get_class_names()), 4)
        print(f"Class names: {dataset.get_class_names()}")
        self.assertEqual(len(dataset.get_class_names()), 4)
        print(f"Dataset length: {len(dataset)}")
        self.assertEqual(len(dataset), 1000 * 4) # Assuming 1000 images per class
        
        # Test getting samples if dataset is not empty
        if len(dataset) > 0:
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                self.assertIsNotNone(sample)
    
    def test_image_dataset(self):
        """Test the image dataset."""
        dataset = AnimalImageDataset(self.data_path)
        
        # Test dataset properties
        self.assertIsInstance(dataset, AnimalImageDataset)
        self.assertGreaterEqual(len(dataset), 0)
        self.assertIsInstance(dataset.get_num_classes(), int)
        self.assertIsInstance(dataset.get_class_names(), list)
        
        # Test getting a sample if dataset is not empty
        if len(dataset) > 0:
            image, category = dataset[0]
            self.assertIsInstance(image, torch.Tensor)
            self.assertEqual(len(image.shape), 3)  # Should be (C, H, W)
            self.assertIsInstance(category, int)
            self.assertGreaterEqual(category, 0)
            self.assertLess(category, dataset.get_num_classes())
    
    def test_dataloader_integration(self):
        """Test the image dataset with DataLoader."""
        dataset = AnimalImageDataset(self.data_path)
        
        if len(dataset) > 0:
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            
            # Get one batch
            for images, labels in dataloader:
                self.assertIsInstance(images, torch.Tensor)
                self.assertIsInstance(labels, torch.Tensor)
                self.assertEqual(len(images.shape), 4)  # Should be (B, C, H, W)
                self.assertEqual(len(labels.shape), 1)  # Should be (B,)
                self.assertEqual(images.shape[0], labels.shape[0])  # Batch sizes should match
                break


if __name__ == "__main__":
    unittest.main()