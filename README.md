# AI Acne Identification & Treatment Recommendation System

An intelligent system that identifies acne lesions on facial images and provides personalized treatment recommendations using advanced computer vision and machine learning.

## 🚀 Features

- **Real-time Acne Detection**: Identifies acne lesions with high accuracy using YOLOv8
- **Visual Detection Results**: Draws bounding boxes around detected acne lesions
- **Severity Assessment**: Evaluates acne severity (mild, moderate, severe) based on lesion count
- **Treatment Recommendations**: Provides personalized treatment suggestions based on severity
- **Image Enhancement**: Automatically enhances images for better detection accuracy
- **Multi-scale Detection**: Uses multiple image scales to detect both large and small lesions
- **Web Interface**: User-friendly web application for easy interaction
- **High Confidence Scores**: Achieves 60-80% confidence in acne detection

## 📊 Model Performance

- **Architecture**: YOLOv8s (small model for optimal balance of speed and accuracy)
- **Training**: 100 epochs with extensive data augmentation
- **Precision**: 59.9% (accuracy of detections)
- **Recall**: 62.1% (ability to find all acne lesions)
- **mAP50**: 59.1% (mean Average Precision at IoU 0.5)
- **Confidence Range**: 60-80% for detected lesions

## 📁 Dataset

This project uses the [Acne Dataset in YOLOv8 Format](https://www.kaggle.com/datasets/osmankagankurnaz/acne-dataset-in-yolov8-format) from Kaggle, which contains:
- Single "acne" class with bounding box annotations
- High-quality facial images in YOLOv8 format
- Diverse skin tones and lighting conditions
- Ready-to-use training/validation/test splits

## 🏗️ Project Structure

```
belle/
├── app/                           # Main application code
│   ├── models/                   # ML model files and training scripts
│   │   ├── acne_detector.pt      # Trained YOLOv8 model (22.5 MB)
│   │   └── train.py              # Model training script
│   ├── api/                      # FastAPI endpoints
│   │   └── main.py               # Main API routes
│   ├── services/                 # Business logic
│   │   ├── acne_detector.py      # Core detection service
│   │   └── simple_treatment_recommender.py  # Treatment recommendations
│   └── utils/                    # Utility functions
│       ├── image_processor.py    # Image processing and visualization
│       └── dataset_downloader.py # Dataset download utility
├── data/                         # Dataset and configuration
│   └── acne_dataset.yaml         # YOLO dataset configuration
├── runs/                         # Training run results
│   └── train/                    # Training outputs and metrics
├── static/                       # Frontend assets and result images
├── templates/                    # HTML templates
│   └── index.html                # Main web interface
├── requirements.txt              # Python dependencies
├── train_model.py                # Training wrapper script
└── README.md                     # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (optional, for faster training)
- Kaggle account (for dataset access)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd belle
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Kaggle API** (for dataset download)
   - Get your Kaggle API key from https://www.kaggle.com/account
   - Place `kaggle.json` in your home directory

## 🚀 Usage

### Quick Start (Using Pre-trained Model)
1. **Start the web application**
   ```bash
   python -m uvicorn app.api.main:app --reload
   ```
2. **Open your browser** to `http://localhost:8000`
3. **Upload a facial image** and get instant acne analysis!

### Training Your Own Model
1. **Download the dataset**
   ```bash
   python app/utils/dataset_downloader.py
   ```

2. **Train the model**
   ```bash
   python train_model.py
   ```
   - Training takes 30-60 minutes depending on hardware
   - Model will be saved to `app/models/acne_detector.pt`

3. **Start the application**
   ```bash
   python -m uvicorn app.api.main:app --reload
   ```

## 🔌 API Endpoints

- `POST /analyze` - Upload and analyze facial image for acne detection
- `GET /acne-types` - Get information about acne types
- `GET /health` - Health check endpoint
- `GET /` - Main web interface

### Example API Usage
```python
import requests

# Upload image for analysis
with open('face_image.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/analyze', files={'file': f})
    result = response.json()
    
print(f"Detected {result['detection_count']} acne lesions")
print(f"Severity: {result['severity']}")
print(f"Confidence scores: {result['confidence_scores']}")
```

## 🎯 Key Features Explained

### Detection Process
1. **Image Upload**: User uploads facial image via web interface
2. **Image Enhancement**: Automatic contrast and quality improvement
3. **Multi-scale Detection**: Model analyzes image at multiple scales
4. **Bounding Box Drawing**: Visual indicators around detected acne
5. **Severity Assessment**: Classification as mild/moderate/severe
6. **Treatment Recommendations**: Personalized suggestions based on severity

### Model Architecture
- **Base Model**: YOLOv8s (small variant for speed/accuracy balance)
- **Input Size**: 640x640 pixels (with multi-scale support)
- **Confidence Threshold**: 0.3 (optimized for recall)
- **NMS IoU**: 0.4 (non-maximum suppression)

## 🔧 Technical Details

### Training Configuration
- **Epochs**: 100
- **Batch Size**: 8
- **Learning Rate**: 0.01
- **Data Augmentation**: Mixup, Copy-paste, HSV adjustments, rotation, scaling
- **Optimizer**: AdamW with cosine learning rate scheduling

### Performance Optimizations
- **Multi-scale Inference**: Detects both large and small lesions
- **Image Enhancement**: Preprocessing for better detection
- **Confidence Tuning**: Optimized thresholds for real-world usage
- **GPU Acceleration**: CUDA support for faster inference

## 🐛 Troubleshooting

### Common Issues
1. **"Model not found" error**
   - Ensure `app/models/acne_detector.pt` exists
   - Run training if model is missing

2. **Low detection accuracy**
   - Check image quality and lighting
   - Ensure face is clearly visible
   - Try different angles or lighting

3. **GPU not being used**
   - Install CUDA-enabled PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
   - Verify CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`

## 📈 Future Improvements

- [ ] Support for multiple acne types (blackheads, whiteheads, etc.)
- [ ] Real-time video analysis
- [ ] Mobile app integration
- [ ] Advanced treatment recommendations
- [ ] Progress tracking over time
- [ ] Integration with dermatology databases

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection framework
- [Kaggle Acne Dataset](https://www.kaggle.com/datasets/osmankagankurnaz/acne-dataset-in-yolov8-format) for training data
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
