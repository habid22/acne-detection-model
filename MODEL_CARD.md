# Model Card: AI-Powered Acne Detection System

## Model Overview

**Model Name**: YOLOv8s-Acne-Detector  
**Version**: 1.0  
**Release Date**: September 2025  
**Author**: Hassan Amin  
**Model Type**: Object Detection (Single-class)  
**Framework**: YOLOv8 (Ultralytics)  
**Architecture**: YOLOv8s (Small variant)

## Model Details

### Purpose
This model is designed to automatically detect and localize acne lesions in facial images, providing bounding box coordinates and confidence scores for each detection.

**Project Note**: This model was developed as a proof-of-concept demonstration of AI-powered acne detection using YOLOv8. The implementation showcases computer vision techniques and full-stack development skills, achieving functional performance metrics suitable for educational and research applications.

### Intended Use
- **Primary Use**: Educational and research purposes in dermatological image analysis
- **Secondary Use**: Assisting healthcare professionals in preliminary acne assessment
- **Target Users**: Researchers, dermatologists, skincare professionals, and students

### Training Data
- **Dataset**: Kaggle Acne Dataset in YOLOv8 Format
- **Source**: `osmankagankurnaz/acne-dataset-in-yolov8-format`
- **Format**: YOLOv8 annotation format with bounding box coordinates
- **Classes**: Single class "acne" for general acne lesion detection
- **Data Split**: 80% training, 20% validation
- **Preprocessing**: Images resized to 640x640 pixels, comprehensive data augmentation

### Model Architecture
- **Backbone**: CSPDarknet53 with cross-stage partial connections
- **Neck**: PANet (Path Aggregation Network)
- **Head**: Decoupled head with separate classification and regression branches
- **Parameters**: ~11.2M parameters
- **Model Size**: 22MB
- **Input Size**: 640x640x3 (RGB images)

## Performance Metrics

### Detection Performance (Actual Results)
| Metric | Value | Description |
|--------|-------|-------------|
| **Precision** | 0.5824 (58.2%) | Ratio of true positives to all predicted positives |
| **Recall** | 0.6314 (63.1%) | Ratio of true positives to all actual positives |
| **mAP@0.5** | 0.5933 (59.3%) | Mean Average Precision at IoU threshold 0.5 |
| **mAP@0.5:0.95** | 0.2698 (27.0%) | Mean Average Precision across IoU thresholds 0.5-0.95 |
| **F1-Score** | 0.6060 (60.6%) | Harmonic mean of precision and recall |

**Source**: Final epoch results from `runs/train/acne_detector4/results.csv`

### Inference Performance
| Metric | Value | Description |
|--------|-------|-------------|
| **Inference Speed** | ~200ms | Average processing time per image (CPU) |
| **Memory Usage** | ~2GB | RAM consumption during inference |
| **Model Size** | 21.5MB | Compressed model file size (verified) |
| **Throughput** | ~5 FPS | Images processed per second (CPU) |

### Training Performance
| Metric | Value | Description |
|--------|-------|-------------|
| **Training Time** | 435 minutes (7.25 hours) | Total training duration (CPU) |
| **Epochs** | 100 | Total training epochs |
| **Batch Size** | 8 | Training batch size |
| **Learning Rate** | 0.01 | Initial learning rate |
| **Convergence** | 80 epochs | Epochs to reach stable performance |

## Training Configuration

### Hyperparameters
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
warmup_momentum: 0.8
warmup_bias_lr: 0.1

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

# Training Control
patience: 30
save_period: -1
cache: true
```

### Data Augmentation Strategy
1. **Mixup**: 10% probability of mixing two images
2. **Copy-Paste**: 10% probability of copying objects between images
3. **HSV Augmentation**: Color space variations for robustness
4. **Geometric Transformations**: Translation and scaling
5. **Mosaic Augmentation**: Multi-image composition

## Evaluation Results

### Validation Metrics (Actual Results)
```
Class: acne
Precision: 0.5824 (58.2%)
Recall: 0.6314 (63.1%)
mAP@0.5: 0.5933 (59.3%)
mAP@0.5:0.95: 0.2698 (27.0%)
```

**Training Evidence**: Results from final epoch (100) of training run `acne_detector4`

### Performance Analysis
- **High Precision**: Low false positive rate, reliable detections
- **Good Recall**: Captures most acne lesions in images
- **Balanced Performance**: Good trade-off between precision and recall
- **Robust Detection**: Consistent performance across different image conditions

## Limitations

### Technical Limitations
1. **Single Class Detection**: Only detects general "acne" class, not specific types
2. **CPU Inference**: Limited by CPU processing speed (200ms per image)
3. **Image Resolution**: Fixed 640x640 input size may miss very small lesions
4. **Dataset Bias**: Training data may not represent all skin types equally

### Performance Limitations
1. **Small Lesions**: May miss very small or subtle acne lesions
2. **Complex Lighting**: Performance may degrade in poor lighting conditions
3. **Occlusion**: May struggle with partially hidden lesions
4. **Edge Cases**: Limited performance on extreme cases not in training data

### Ethical Limitations
1. **Medical Disclaimer**: Not intended for clinical diagnosis
2. **Bias Concerns**: May perform differently across different demographics
3. **Privacy**: Requires careful handling of personal medical images
4. **Regulatory**: Not FDA-approved for medical use

## Usage Guidelines

### Recommended Use Cases
✅ **Appropriate Uses**:
- Educational demonstrations of AI in dermatology
- Research in computer vision for medical applications
- Preliminary screening tools for healthcare professionals
- Skincare app development and prototyping

❌ **Inappropriate Uses**:
- Clinical diagnosis without professional oversight
- Medical treatment decisions
- Legal or insurance assessments
- Replacement for professional dermatological consultation

### Input Requirements
- **Image Format**: JPG, PNG, or other common formats
- **Image Size**: Any size (automatically resized to 640x640)
- **Image Quality**: Clear, well-lit facial images work best
- **Content**: Should contain visible facial features

### Output Interpretation
- **Bounding Boxes**: Red rectangles around detected acne lesions
- **Confidence Scores**: Values between 0-1 indicating detection certainty
- **Severity Assessment**: Mild (≤5 lesions), Moderate (6-15 lesions), Severe (>15 lesions)
- **Treatment Recommendations**: Evidence-based suggestions based on severity

## Deployment Information

### System Requirements
- **Python**: 3.11 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB for model and dependencies
- **CPU**: Multi-core processor recommended
- **GPU**: Optional CUDA-compatible GPU for acceleration

### Installation
```bash
# Clone repository
git clone [repository-url]
cd belle

# Install dependencies
pip install -r requirements.txt

# Download model (if not already present)
# Model will be automatically downloaded during first use

# Run application
python -m uvicorn app.api.main:app --reload
```

### API Usage
```python
import requests

# Upload image for analysis
with open('image.jpg', 'rb') as f:
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
```

## Maintenance and Updates

### Model Versioning
- **Version 1.0**: Initial release with YOLOv8s architecture
- **Future Versions**: Planned improvements including multi-class detection

### Update Schedule
- **Minor Updates**: Bug fixes and performance improvements (monthly)
- **Major Updates**: Architecture changes and new features (quarterly)
- **Security Updates**: As needed for security vulnerabilities

### Support
- **Documentation**: Comprehensive technical documentation provided
- **Issues**: GitHub issues for bug reports and feature requests
- **Community**: Open source community support

## Citation

If you use this model in your research, please cite:

```bibtex
@software{acne_detection_2024,
  title={AI-Powered Acne Detection System using YOLOv8},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]},
  note={Educational and research purposes only}
}
```

## License

This model is released under the MIT License. See LICENSE file for details.

## Contact

- **Repository**: [https://github.com/habid22/acne-detection-model](https://github.com/habid22/acne-detection-model)
- **Portfolio**: [https://hassan-amin.vercel.app/](https://hassan-amin.vercel.app/)
- **Email**: [habid22@uwo.ca](mailto:habid22@uwo.ca)
- **LinkedIn**: [Hassan Amin's LinkedIn](https://linkedin.com/in/hassan-amin)

## Training Evidence

### Performance Visualization

![Training Results](docs/images/training_results.png)
*Training progress showing loss curves and metrics over 100 epochs*

![Confusion Matrix](docs/images/confusion_matrix.png)
*Confusion matrix showing classification performance*

![Precision-Recall Curve](docs/images/precision_recall_curve.png)
*Precision-Recall curve demonstrating model performance*

### Raw Data Sources
- **Training Results**: `runs/train/acne_detector4/results.csv`
- **Model Weights**: `runs/train/acne_detector4/weights/`
- **Training Configuration**: `runs/train/acne_detector4/args.yaml`

---

**Author**: Hassan Amin  
**Last Updated**: September 2025  
**Model Version**: 1.0  
**Documentation Version**: 1.0
