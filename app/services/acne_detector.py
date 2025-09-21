"""
Acne Detection Service using YOLOv8
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any
import os
from pathlib import Path

class AcneDetector:
    """
    Service for detecting and classifying acne lesions in facial images
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the acne detector
        
        Args:
            model_path: Path to the trained YOLO model
        """
        self.model_path = model_path or "app/models/acne_detector.pt"
        self.model = None
        self.acne_classes = {
            0: "acne"  # General acne detection
        }
        
        # Load model if it exists
        if os.path.exists(self.model_path):
            self.load_model()
    
    def load_model(self):
        """Load the trained YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to pretrained YOLOv8 model
            self.model = YOLO('yolov8n.pt')
    
    def detect_acne(self, image: np.ndarray, confidence_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Detect acne lesions in the input image
        
        Args:
            image: Input image as numpy array
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            Dictionary containing detection results
        """
        if self.model is None:
            self.load_model()
        
        # Run inference with lower NMS threshold for more detections
        # Also try different image sizes for better detection
        results = self.model(image, conf=confidence_threshold, iou=0.4, imgsz=640)
        
        # Try with smaller image size for smaller lesions
        if len(results[0].boxes) < 5:  # If we don't have many detections
            results_small = self.model(image, conf=confidence_threshold*0.8, iou=0.3, imgsz=416)
            # Combine results
            if len(results_small[0].boxes) > len(results[0].boxes):
                results = results_small
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Extract detection information
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    detection = {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(confidence),
                        "class": self.acne_classes.get(class_id, "unknown"),
                        "class_id": class_id
                    }
                    detections.append(detection)
        
        return {
            "detections": detections,
            "total_detections": len(detections),
            "image_shape": image.shape
        }
    
    def assess_severity(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the severity of acne based on detection results
        
        Args:
            detection_results: Results from detect_acne method
            
        Returns:
            Severity assessment
        """
        detections = detection_results.get("detections", [])
        
        if not detections:
            return {
                "level": "none",
                "score": 0,
                "description": "No acne detected"
            }
        
        # Count detections by class
        class_counts = {}
        total_confidence = 0
        
        for detection in detections:
            acne_class = detection["class"]
            confidence = detection["confidence"]
            
            class_counts[acne_class] = class_counts.get(acne_class, 0) + 1
            total_confidence += confidence
        
        # Calculate severity score
        total_detections = len(detections)
        avg_confidence = total_confidence / total_detections if total_detections > 0 else 0
        
        # Weight different acne types (simplified for general acne detection)
        severity_weights = {
            "acne": 1  # All acne lesions weighted equally
        }
        
        weighted_score = 0
        for acne_class, count in class_counts.items():
            weight = severity_weights.get(acne_class, 1)
            weighted_score += count * weight
        
        # Determine severity level (adjusted for general acne detection)
        if weighted_score <= 5:
            level = "mild"
        elif weighted_score <= 15:
            level = "moderate"
        else:
            level = "severe"
        
        return {
            "level": level,
            "score": weighted_score,
            "total_detections": total_detections,
            "class_counts": class_counts,
            "average_confidence": avg_confidence,
            "description": f"Acne severity: {level.title()}"
        }
    
    def generate_summary(self, detection_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of the detection results
        
        Args:
            detection_results: Results from detect_acne method
            
        Returns:
            Summary string
        """
        detections = detection_results.get("detections", [])
        
        if not detections:
            return "No acne lesions detected in the image."
        
        # Count by class
        class_counts = {}
        for detection in detections:
            acne_class = detection["class"]
            class_counts[acne_class] = class_counts.get(acne_class, 0) + 1
        
        # Generate summary
        summary_parts = []
        for acne_class, count in class_counts.items():
            if count == 1:
                summary_parts.append(f"1 {acne_class.replace('_', ' ')}")
            else:
                summary_parts.append(f"{count} {acne_class.replace('_', ' ')}")
        
        summary = f"Detected {len(detections)} acne lesions: {', '.join(summary_parts)}."
        
        return summary
    
    def train_model(self, dataset_path: str, epochs: int = 100):
        """
        Train the acne detection model
        
        Args:
            dataset_path: Path to the dataset
            epochs: Number of training epochs
        """
        if self.model is None:
            self.model = YOLO('yolov8n.pt')
        
        # Train the model
        results = self.model.train(
            data=dataset_path,
            epochs=epochs,
            imgsz=640,
            batch=16,
            name='acne_detector'
        )
        
        # Save the trained model
        self.model.save(self.model_path)
        
        return results
