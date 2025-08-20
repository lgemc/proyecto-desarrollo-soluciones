import os
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import Dataset


class AnimalMetadataDataset(Dataset):
    """Dataset that returns metadata about animal images."""
    
    CATEGORIES = {
        'Buffalo': 0,
        'Elephant': 1,
        'Rhino': 2,
        'Zebra': 3
    }
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to the data directory containing animal folders
        """
        self.data_dir = Path(data_dir)
        self.samples = []
        
        # Collect all image files
        for animal_name, class_id in self.CATEGORIES.items():
            animal_dir = self.data_dir / animal_name
            if animal_dir.exists():
                for img_file in sorted(animal_dir.glob('*.jpg')):
                    # Store relative path from data folder
                    relative_path = str(img_file.relative_to(self.data_dir))
                    self.samples.append({
                        'class': class_id,
                        'description': animal_name,
                        'image_path': relative_path
                    })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get item at index.
        
        Returns:
            Dictionary with keys: 'class' (int), 'description' (str), 'image_path' (str)
        """
        return self.samples[idx].copy()
    
    def get_num_classes(self) -> int:
        """Get the number of unique classes."""
        return len(self.CATEGORIES)
    
    def get_class_names(self) -> List[str]:
        """Get list of class names ordered by class ID."""
        return sorted(self.CATEGORIES.keys(), key=lambda x: self.CATEGORIES[x])