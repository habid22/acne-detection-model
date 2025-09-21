"""
Training script for acne detection model
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import argparse

def create_dataset_config(dataset_path: str, output_path: str = "data/acne_dataset.yaml"):
    """
    Create YAML configuration file for the dataset
    
    Args:
        dataset_path: Path to the dataset directory
        output_path: Output path for the YAML file
    """
    config = {
        'path': dataset_path,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 6,  # Number of classes
        'names': [
            'blackheads',
            'dark_spots',
            'nodules', 
            'papules',
            'pustules',
            'whiteheads'
        ]
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write YAML file
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Dataset configuration saved to {output_path}")
    return output_path

def download_kaggle_dataset(kaggle_username: str, kaggle_key: str):
    """
    Download dataset from Kaggle
    
    Args:
        kaggle_username: Kaggle username
        kaggle_key: Kaggle API key
    """
    try:
        from ..utils.dataset_downloader import download_acne_dataset
        
        dataset_path, config_path = download_acne_dataset(
            kaggle_username=kaggle_username,
            kaggle_key=kaggle_key
        )
        
        if dataset_path:
            print(f"Dataset downloaded to: {dataset_path}")
            return dataset_path, config_path
        else:
            return None, None
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None, None

def train_model(dataset_config: str, epochs: int = 100, imgsz: int = 640, batch: int = 16):
    """
    Train the acne detection model
    
    Args:
        dataset_config: Path to dataset YAML configuration
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
    """
    # Initialize YOLO model - using small model for better accuracy
    model = YOLO('yolov8s.pt')  # Small model for better accuracy than nano
    
    # Train the model with enhanced parameters for better confidence
    results = model.train(
        data=dataset_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name='acne_detector',
        project='runs/train',
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        patience=30,     # Increased patience for better convergence
        device='cpu',    # Use 'cuda' if GPU available
        workers=4,
        cache=True,
        augment=True,
        mixup=0.15,      # Increased mixup for better generalization
        copy_paste=0.15, # Increased copy-paste augmentation
        hsv_h=0.015,     # Color augmentation
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,    # Rotation augmentation
        translate=0.1,   # Translation augmentation
        scale=0.5,       # Scale augmentation
        shear=2.0,       # Shear augmentation
        perspective=0.0, # Perspective augmentation
        flipud=0.0,      # Vertical flip
        fliplr=0.5,      # Horizontal flip
        mosaic=1.0,      # Mosaic augmentation
        lr0=0.01,        # Learning rate
        lrf=0.01,        # Final learning rate
        momentum=0.937,  # SGD momentum
        weight_decay=0.0005,  # Weight decay
        warmup_epochs=3, # Warmup epochs
        warmup_momentum=0.8,
        warmup_bias_lr=0.1
    )
    
    # Save the best model
    best_model_path = f"runs/train/acne_detector/weights/best.pt"
    final_model_path = "app/models/acne_detector.pt"
    
    # Create models directory if it doesn't exist
    os.makedirs("app/models", exist_ok=True)
    
    # Copy best model to final location
    if os.path.exists(best_model_path):
        import shutil
        shutil.copy2(best_model_path, final_model_path)
        print(f"Best model saved to {final_model_path}")
    
    return results

def validate_model(model_path: str, dataset_config: str):
    """
    Validate the trained model
    
    Args:
        model_path: Path to the trained model
        dataset_config: Path to dataset configuration
    """
    model = YOLO(model_path)
    
    # Run validation
    results = model.val(data=dataset_config)
    
    print("Validation Results:")
    print(f"mAP50: {results.box.map50:.3f}")
    print(f"mAP50-95: {results.box.map:.3f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train acne detection model')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset directory')
    parser.add_argument('--kaggle-username', type=str, help='Kaggle username')
    parser.add_argument('--kaggle-key', type=str, help='Kaggle API key')
    parser.add_argument('--zip-path', type=str, help='Path to manually downloaded zip file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--validate-only', action='store_true', help='Only validate existing model')
    
    args = parser.parse_args()
    
    if args.validate_only:
        # Validate existing model
        model_path = "app/models/acne_detector.pt"
        dataset_config = "data/acne_dataset.yaml"
        
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return
        
        if not os.path.exists(dataset_config):
            print(f"Dataset config not found at {dataset_config}")
            return
        
        validate_model(model_path, dataset_config)
        return
    
    # Determine dataset path
    if args.dataset_path:
        dataset_path = args.dataset_path
        dataset_config = create_dataset_config(dataset_path)
    elif args.kaggle_username and args.kaggle_key:
        # Download from Kaggle
        dataset_path, dataset_config = download_kaggle_dataset(
            args.kaggle_username, 
            args.kaggle_key
        )
        if not dataset_path:
            return
    elif args.zip_path:
        # Manual download
        from ..utils.dataset_downloader import download_acne_dataset
        dataset_path, dataset_config = download_acne_dataset(zip_path=args.zip_path)
        if not dataset_path:
            return
    else:
        print("Please provide one of the following:")
        print("  --dataset-path: Path to existing dataset")
        print("  --kaggle-username and --kaggle-key: Kaggle credentials")
        print("  --zip-path: Path to manually downloaded zip file")
        return
    
    # Train the model
    print("Starting model training...")
    results = train_model(
        dataset_config, 
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch
    )
    
    # Validate the model
    print("Validating trained model...")
    validate_model("app/models/acne_detector.pt", dataset_config)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
