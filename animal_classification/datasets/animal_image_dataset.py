import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
from .animal_metadata_dataset import AnimalMetadataDataset


class AnimalImageDataset(Dataset):
    """Dataset that returns processed images and their categories."""
    
    def __init__(
        self, 
        data_dir: str = 'data',
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize the image dataset.
        
        Args:
            data_dir: Path to the data directory containing animal folders
            transform: Optional torchvision transforms to apply to images
            target_size: Target size for image resizing (height, width)
        """
        # Use the metadata dataset to get file information
        self.metadata_dataset = AnimalMetadataDataset(data_dir)
        self.data_dir = Path(data_dir)
        
        # Set up default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet standards
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        return len(self.metadata_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get image and category at index.
        
        Returns:
            Tuple of (image_tensor, category_id)
        """
        # Get metadata
        metadata = self.metadata_dataset[idx]
        
        # Load image
        img_path = self.data_dir / metadata['image_path']
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Return image tensor and category
        return image, metadata['class']
    
    def get_num_classes(self) -> int:
        """Get the number of unique classes."""
        return self.metadata_dataset.get_num_classes()
    
    def get_class_names(self) -> list:
        """Get list of class names ordered by class ID."""
        return self.metadata_dataset.get_class_names()