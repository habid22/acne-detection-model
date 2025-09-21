"""
Dataset downloader for Kaggle acne dataset
"""

import os
import zipfile
import shutil
from pathlib import Path
import requests
import json
from typing import Optional

class KaggleDatasetDownloader:
    """
    Utility class for downloading and setting up the Kaggle acne dataset
    """
    
    def __init__(self, kaggle_username: str, kaggle_key: str):
        """
        Initialize the downloader with Kaggle credentials
        
        Args:
            kaggle_username: Your Kaggle username
            kaggle_key: Your Kaggle API key
        """
        self.kaggle_username = kaggle_username
        self.kaggle_key = kaggle_key
        self.dataset_name = "osmankagankurnaz/acne-dataset-in-yolov8-format"
        self.dataset_path = "data/acne_dataset"
        
    def setup_kaggle_credentials(self):
        """Set up Kaggle API credentials"""
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        
        kaggle_config = {
            "username": self.kaggle_username,
            "key": self.kaggle_key
        }
        
        with open(kaggle_dir / "kaggle.json", "w") as f:
            json.dump(kaggle_config, f)
        
        # Set proper permissions
        os.chmod(kaggle_dir / "kaggle.json", 0o600)
        
        print("Kaggle credentials configured successfully")
    
    def download_dataset(self) -> str:
        """
        Download the acne dataset from Kaggle
        
        Returns:
            Path to the downloaded dataset
        """
        try:
            import kaggle
            
            # Set up credentials
            self.setup_kaggle_credentials()
            
            # Create data directory
            os.makedirs("data", exist_ok=True)
            
            # Download dataset
            print(f"Downloading dataset: {self.dataset_name}")
            kaggle.api.dataset_download_files(
                self.dataset_name,
                path="data",
                unzip=True
            )
            
            # Find the extracted directory
            data_dir = Path("data")
            extracted_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
            
            if extracted_dirs:
                dataset_dir = extracted_dirs[0]
                print(f"Dataset downloaded to: {dataset_dir}")
                return str(dataset_dir)
            else:
                raise Exception("Dataset extraction failed")
                
        except ImportError:
            print("Kaggle package not installed. Installing...")
            os.system("pip install kaggle")
            return self.download_dataset()
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None
    
    def download_manual(self, zip_path: str) -> str:
        """
        Manual download method - user downloads zip file manually
        
        Args:
            zip_path: Path to the manually downloaded zip file
            
        Returns:
            Path to the extracted dataset
        """
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Zip file not found: {zip_path}")
        
        # Create data directory
        os.makedirs("data", exist_ok=True)
        
        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data")
        
        # Find extracted directory
        data_dir = Path("data")
        extracted_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        
        if extracted_dirs:
            dataset_dir = extracted_dirs[0]
            print(f"Dataset extracted to: {dataset_dir}")
            return str(dataset_dir)
        else:
            raise Exception("Dataset extraction failed")
    
    def organize_dataset(self, dataset_path: str) -> str:
        """
        Organize the dataset into proper YOLO structure
        
        Args:
            dataset_path: Path to the downloaded dataset
            
        Returns:
            Path to the organized dataset
        """
        dataset_path = Path(dataset_path)
        organized_path = Path("data/acne_dataset_organized")
        
        # Create organized structure
        for split in ['train', 'val', 'test']:
            (organized_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (organized_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Find existing structure
        if (dataset_path / 'train').exists():
            # Already organized
            print("Dataset is already in YOLO format")
            return str(dataset_path)
        
        # Look for images and labels
        images_dir = None
        labels_dir = None
        
        for item in dataset_path.rglob("*"):
            if item.is_dir():
                if 'image' in item.name.lower():
                    images_dir = item
                elif 'label' in item.name.lower():
                    labels_dir = item
        
        if not images_dir or not labels_dir:
            print("Warning: Could not find standard image/label directories")
            print("Please check the dataset structure manually")
            return str(dataset_path)
        
        # Copy files to organized structure
        # This is a simplified version - you may need to adjust based on actual structure
        print("Organizing dataset structure...")
        
        return str(dataset_path)
    
    def create_dataset_config(self, dataset_path: str) -> str:
        """
        Create YAML configuration file for the dataset
        
        Args:
            dataset_path: Path to the organized dataset
            
        Returns:
            Path to the YAML configuration file
        """
        import yaml
        
        # Check what classes are in the dataset
        classes = self._detect_classes(dataset_path)
        
        config = {
            'path': str(Path(dataset_path).absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(classes),
            'names': classes
        }
        
        config_path = "data/acne_dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Dataset configuration saved to: {config_path}")
        print(f"Detected classes: {classes}")
        
        return config_path
    
    def _detect_classes(self, dataset_path: str) -> list:
        """
        Detect classes from label files
        
        Args:
            dataset_path: Path to the dataset
            
        Returns:
            List of class names
        """
        class_ids = set()
        dataset_path = Path(dataset_path)
        
        # Look for label files
        for label_file in dataset_path.rglob("*.txt"):
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            class_ids.add(class_id)
            except:
                continue
        
        # Map class IDs to names (you may need to adjust this)
        class_mapping = {
            0: "blackheads",
            1: "whiteheads", 
            2: "papules",
            3: "pustules",
            4: "nodules",
            5: "dark_spots"
        }
        
        classes = [class_mapping.get(i, f"class_{i}") for i in sorted(class_ids)]
        return classes

def download_acne_dataset(kaggle_username: str = None, kaggle_key: str = None, zip_path: str = None):
    """
    Main function to download and set up the acne dataset
    
    Args:
        kaggle_username: Kaggle username (optional)
        kaggle_key: Kaggle API key (optional)
        zip_path: Path to manually downloaded zip file (optional)
    """
    if zip_path:
        # Manual download
        downloader = KaggleDatasetDownloader("", "")
        dataset_path = downloader.download_manual(zip_path)
    else:
        # Kaggle API download
        if not kaggle_username or not kaggle_key:
            print("Please provide Kaggle username and API key")
            print("You can get your API key from: https://www.kaggle.com/settings")
            return None
        
        downloader = KaggleDatasetDownloader(kaggle_username, kaggle_key)
        dataset_path = downloader.download_dataset()
    
    if dataset_path:
        # Organize dataset
        organized_path = downloader.organize_dataset(dataset_path)
        
        # Create configuration
        config_path = downloader.create_dataset_config(organized_path)
        
        print(f"\nDataset setup complete!")
        print(f"Dataset path: {organized_path}")
        print(f"Config path: {config_path}")
        
        return organized_path, config_path
    
    return None, None

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Download acne dataset')
    parser.add_argument('--kaggle-username', type=str, help='Kaggle username')
    parser.add_argument('--kaggle-key', type=str, help='Kaggle API key')
    parser.add_argument('--zip-path', type=str, help='Path to manually downloaded zip file')
    
    args = parser.parse_args()
    
    download_acne_dataset(
        kaggle_username=args.kaggle_username,
        kaggle_key=args.kaggle_key,
        zip_path=args.zip_path
    )
