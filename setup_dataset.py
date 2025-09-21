"""
Setup script for downloading and preparing the acne dataset
"""

import os
import sys
from pathlib import Path

def main():
    print("🔬 AI Acne Identification - Dataset Setup")
    print("=" * 50)
    
    print("\n📋 Dataset Options:")
    print("1. Download from Kaggle (requires API key)")
    print("2. Manual download (you download zip file)")
    print("3. Use existing dataset")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        setup_kaggle_download()
    elif choice == "2":
        setup_manual_download()
    elif choice == "3":
        setup_existing_dataset()
    else:
        print("Invalid choice. Please run the script again.")
        return

def setup_kaggle_download():
    """Setup Kaggle API download"""
    print("\n🔑 Kaggle API Setup")
    print("You need a Kaggle account and API key.")
    print("Get your API key from: https://www.kaggle.com/settings")
    
    username = input("Enter your Kaggle username: ").strip()
    api_key = input("Enter your Kaggle API key: ").strip()
    
    if not username or not api_key:
        print("❌ Username and API key are required")
        return
    
    print("\n📥 Downloading dataset...")
    try:
        from app.utils.dataset_downloader import download_acne_dataset
        
        dataset_path, config_path = download_acne_dataset(
            kaggle_username=username,
            kaggle_key=api_key
        )
        
        if dataset_path:
            print(f"✅ Dataset downloaded successfully!")
            print(f"📁 Dataset path: {dataset_path}")
            print(f"⚙️  Config path: {config_path}")
            show_next_steps()
        else:
            print("❌ Failed to download dataset")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def setup_manual_download():
    """Setup manual download"""
    print("\n📥 Manual Download Setup")
    print("1. Go to: https://www.kaggle.com/datasets/osmankagankurnaz/acne-dataset-in-yolov8-format/data")
    print("2. Click 'Download' button")
    print("3. Download the zip file (36 MB)")
    print("4. Save it to your project directory")
    
    zip_path = input("\nEnter path to downloaded zip file: ").strip()
    
    if not os.path.exists(zip_path):
        print("❌ File not found. Please check the path.")
        return
    
    print("\n📦 Extracting dataset...")
    try:
        from app.utils.dataset_downloader import download_acne_dataset
        
        dataset_path, config_path = download_acne_dataset(zip_path=zip_path)
        
        if dataset_path:
            print(f"✅ Dataset extracted successfully!")
            print(f"📁 Dataset path: {dataset_path}")
            print(f"⚙️  Config path: {config_path}")
            show_next_steps()
        else:
            print("❌ Failed to extract dataset")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def setup_existing_dataset():
    """Setup existing dataset"""
    print("\n📁 Existing Dataset Setup")
    dataset_path = input("Enter path to your dataset directory: ").strip()
    
    if not os.path.exists(dataset_path):
        print("❌ Directory not found. Please check the path.")
        return
    
    print("\n⚙️  Creating dataset configuration...")
    try:
        from app.models.train import create_dataset_config
        
        config_path = create_dataset_config(dataset_path)
        
        print(f"✅ Configuration created successfully!")
        print(f"📁 Dataset path: {dataset_path}")
        print(f"⚙️  Config path: {config_path}")
        show_next_steps()
        
    except Exception as e:
        print(f"❌ Error: {e}")

def show_next_steps():
    """Show next steps after dataset setup"""
    print("\n🚀 Next Steps:")
    print("1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n2. Train the model:")
    print("   python app/models/train.py --dataset-path data/acne_dataset")
    print("\n3. Start the web application:")
    print("   uvicorn app.api.main:app --reload")
    print("\n4. Open your browser to: http://localhost:8000")
    
    print("\n📚 Additional Commands:")
    print("• Analyze dataset: jupyter notebook notebooks/data_analysis.ipynb")
    print("• Validate model: python app/models/train.py --validate-only")
    print("• Check dataset: python -c \"from app.utils.dataset_downloader import *; print('Dataset tools loaded')\"")

if __name__ == "__main__":
    main()
