# AI-Powered Acne Detection & Treatment Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-8.3.202-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**An intelligent computer vision system that automatically detects acne lesions in facial images and provides evidence-based treatment recommendations using state-of-the-art deep learning.**

**Author**: Hassan Amin | **Date**: September 2025

[🚀 Quick Start](#-quick-start) • [📊 Performance](#-model-performance) • [🔧 Technical Details](#-technical-specifications) • [📚 Documentation](#-documentation)

</div>

## 🎯 Overview

This project presents a comprehensive AI-powered acne detection system built using YOLOv8 architecture. The system demonstrates functional performance in dermatological image analysis, achieving moderate precision and recall rates for acne lesion detection. The implementation includes a full-stack web application with real-time image processing capabilities and intelligent treatment recommendations.

**Project Note**: This model was developed as a proof-of-concept demonstration of AI-powered acne detection using YOLOv8. The implementation showcases computer vision techniques and full-stack development skills, achieving functional performance metrics suitable for educational and research applications.

### Key Capabilities

- **🔍 Real-time Acne Detection**: Identifies acne lesions with solid accuracy using YOLOv8s architecture
- **📦 Visual Detection Results**: Draws precise bounding boxes around detected acne lesions
- **📊 Severity Assessment**: Evaluates acne severity (mild, moderate, severe) based on lesion count
- **💊 Treatment Recommendations**: Provides evidence-based treatment suggestions based on severity
- **🖼️ Advanced Image Enhancement**: Automatically enhances images for improved detection accuracy
- **📏 Multi-scale Detection**: Uses multiple image scales to detect both large and small lesions
- **🌐 Interactive Web Interface**: User-friendly web application with real-time analysis
- **⚡ Solid Performance**: Achieves 60%+ confidence in acne detection with fast inference

## 📊 Model Performance

### Detection Metrics (Actual Results)
| Metric | Value | Description |
|--------|-------|-------------|
| **Precision** | 58.2% | Accuracy of positive detections |
| **Recall** | 63.1% | Ability to find all acne lesions |
| **mAP@0.5** | 59.3% | Mean Average Precision at IoU 0.5 |
| **mAP@0.5:0.95** | 27.0% | Mean Average Precision across IoU thresholds |
| **F1-Score** | 60.6% | Harmonic mean of precision and recall |

**Source**: Final epoch results from training run `acne_detector4`

### System Performance
| Metric | Value | Description |
|--------|-------|-------------|
| **Inference Speed** | ~200ms | Average processing time per image (CPU) |
| **Model Size** | 21.5MB | Compressed model file size (verified) |
| **Memory Usage** | ~2GB | RAM consumption during inference |
| **Throughput** | ~5 FPS | Images processed per second (CPU) |

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

## 🚀 Quick Start

### Prerequisites
- **Python**: 3.11 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB for model and dependencies
- **CPU**: Multi-core processor recommended
- **GPU**: Optional CUDA-compatible GPU for acceleration

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/habid22/acne-detection-model.git
   cd acne-detection-model
   ```

2. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Kaggle API** (for dataset download)
   ```bash
   # Get your Kaggle API key from https://www.kaggle.com/account
   # Place kaggle.json in your home directory (~/.kaggle/kaggle.json)
   ```

### Usage

#### Option 1: Using Pre-trained Model (Recommended)
```bash
# Start the web application
python -m uvicorn app.api.main:app --reload

# Open your browser to http://localhost:8000
# Upload a facial image and get instant acne analysis!
```

#### Option 2: Train Your Own Model
```bash
# Download the dataset
python app/utils/dataset_downloader.py

# Train the model (takes 30-60 minutes depending on hardware)
python train_model.py

# Start the application
python -m uvicorn app.api.main:app --reload
```

## 🔌 API Documentation

### Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload and analyze facial image for acne detection |
| `GET` | `/acne-types` | Get information about acne types |
| `GET` | `/treatments` | Get treatment recommendations |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/` | Main web interface |

### Example API Usage
```python
import requests

# Upload image for analysis
with open('face_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/upload',
        files={'file': f},
        data={
            'detection_mode': 'standard',
            'confidence_threshold': 0.25
        }
    )
    result = response.json()
    
print(f"Detected {len(result['detections'])} acne lesions")
print(f"Severity: {result['severity']}")
print(f"Treatment: {result['treatment_recommendation']}")
print(f"Result image: {result['result_image']}")
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

## 🔧 Technical Specifications

### Model Architecture
- **Base Model**: YOLOv8s (small variant for optimal speed/accuracy balance)
- **Parameters**: ~11.2M parameters
- **Model Size**: 22MB
- **Input Size**: 640x640x3 (RGB images)
- **Backbone**: CSPDarknet53 with cross-stage partial connections
- **Neck**: PANet (Path Aggregation Network)
- **Head**: Decoupled head with separate classification and regression branches

### Training Configuration
```yaml
# Model Configuration
model: yolov8s.pt
epochs: 100
batch_size: 8
imgsz: 640
device: cpu

# Optimization
optimizer: AdamW
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3

# Data Augmentation
mixup: 0.1
copy_paste: 0.1
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
mosaic: 1.0
```

### Performance Optimizations
- **Multi-scale Detection**: Analyzes images at multiple scales (640x640, 416x416, 832x832)
- **Advanced Image Enhancement**: CLAHE, sharpening, gamma correction
- **Confidence Tuning**: Optimized thresholds for real-world usage
- **Non-Maximum Suppression**: Removes duplicate detections
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

## 📚 Documentation

- **[Technical Report](TECHNICAL_REPORT.md)**: Comprehensive technical documentation
- **[Model Card](MODEL_CARD.md)**: Detailed model specifications and performance metrics
- **[API Documentation](#-api-documentation)**: Complete API reference
- **[Troubleshooting](#-troubleshooting)**: Common issues and solutions

## 🚀 Future Enhancements

### Planned Features
- [ ] **Multi-class Detection**: Separate acne types (papules, pustules, nodules, cysts)
- [ ] **Severity Quantification**: Precise lesion counting and measurement
- [ ] **Real-time Video Analysis**: Live video stream processing
- [ ] **Mobile Application**: Native iOS and Android apps
- [ ] **3D Analysis**: Depth-aware lesion detection
- [ ] **Progress Tracking**: Longitudinal acne monitoring over time
- [ ] **Federated Learning**: Privacy-preserving model updates
- [ ] **Integration**: Dermatology database connections

### Research Opportunities
- [ ] **Bias Mitigation**: Improving performance across diverse populations
- [ ] **Explainable AI**: Visual explanations for detection decisions
- [ ] **Active Learning**: Continuous model improvement with user feedback
- [ ] **Edge Deployment**: Optimized models for mobile and IoT devices

## ⚠️ Important Disclaimers

### Medical Disclaimer
**This system is designed for educational and research purposes only. It should not replace professional medical diagnosis or treatment. Users should consult qualified healthcare professionals for medical advice.**

### Limitations
- Single-class detection (general "acne" only)
- Performance may vary across different skin types and lighting conditions
- Not FDA-approved for clinical use
- Requires clear, well-lit facial images for optimal performance

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** - State-of-the-art object detection framework
- **[Kaggle Acne Dataset](https://www.kaggle.com/datasets/osmankagankurnaz/acne-dataset-in-yolov8-format)** - High-quality training data
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern, fast web framework for building APIs
- **[OpenCV](https://opencv.org/)** - Computer vision library for image processing
- **[PyTorch](https://pytorch.org/)** - Deep learning framework

## 📞 Contact

- **Author**: Hassan Amin
- **Repository**: [https://github.com/habid22/acne-detection-model](https://github.com/habid22/acne-detection-model)
- **Portfolio**: [https://hassan-amin.vercel.app/](https://hassan-amin.vercel.app/)
- **Email**: [habid22@uwo.ca](mailto:habid22@uwo.ca)
- **LinkedIn**: [Hassan Amin's LinkedIn](https://linkedin.com/in/hassan-amin)

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

Made with ❤️ for the AI and medical research community

</div>
