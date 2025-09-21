"""
Simple training script for acne detection model
"""

import os
import sys
from pathlib import Path

def main():
    print("🔬 AI Acne Detection - Model Training")
    print("=" * 50)
    
    # Check if dataset exists
    dataset_path = "data/acne-data/data-2"
    config_path = "data/acne_dataset.yaml"
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset not found!")
        print(f"Expected path: {dataset_path}")
        print("Please make sure you have downloaded the dataset.")
        return
    
    if not os.path.exists(config_path):
        print("❌ Dataset configuration not found!")
        print(f"Expected path: {config_path}")
        return
    
    print(f"✅ Dataset found: {dataset_path}")
    print(f"✅ Configuration found: {config_path}")
    
    # Check dataset structure
    required_dirs = ["train/images", "train/labels", "valid/images", "valid/labels"]
    for dir_path in required_dirs:
        full_path = os.path.join(dataset_path, dir_path)
        if not os.path.exists(full_path):
            print(f"❌ Missing directory: {full_path}")
            return
        else:
            file_count = len([f for f in os.listdir(full_path) if f.endswith(('.jpg', '.txt'))])
            print(f"✅ {dir_path}: {file_count} files")
    
    print("\n🚀 Starting model training...")
    print("This may take 30-60 minutes depending on your hardware.")
    
    # Import and run training
    try:
        from app.models.train import train_model, validate_model
        
        # Train the model
        results = train_model(
            dataset_config=config_path,
            epochs=100,  # Increased for better accuracy
            imgsz=640,
            batch=8     # Reduced for memory efficiency
        )
        
        print("\n✅ Training completed!")
        
        # Validate the model
        print("\n📊 Validating model...")
        model_path = "app/models/acne_detector.pt"
        if os.path.exists(model_path):
            validate_model(model_path, config_path)
        else:
            print("❌ Trained model not found!")
        
        print("\n🎉 Model training and validation complete!")
        print("\nNext steps:")
        print("1. Start the web application:")
        print("   uvicorn app.api.main:app --reload")
        print("2. Open your browser to: http://localhost:8000")
        print("3. Upload a facial image to test the model!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("Please check the error and try again.")

if __name__ == "__main__":
    main()
