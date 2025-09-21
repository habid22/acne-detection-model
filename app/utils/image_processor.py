"""
Image Processing Utilities
"""

import cv2
import numpy as np
from PIL import Image
import io
from typing import Union, Tuple

class ImageProcessor:
    """
    Utility class for image preprocessing and processing
    """
    
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        """
        Initialize image processor
        
        Args:
            target_size: Target size for resizing images (width, height)
        """
        self.target_size = target_size
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """
        Preprocess image data for model input
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Preprocessed image as numpy array
        """
        # Convert bytes to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        image = np.array(pil_image)
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    
    def resize_image(self, image: np.ndarray, size: Tuple[int, int] = None) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: Input image
            size: Target size (width, height)
            
        Returns:
            Resized image
        """
        if size is None:
            size = self.target_size
        
        return cv2.resize(image, size)
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better detection
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def detect_face(self, image: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Detect if image contains a face
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (has_face, face_region)
        """
        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Get the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Extract face region
            face_region = image[y:y+h, x:x+w]
            
            return True, face_region
        
        return False, image
    
    def validate_image(self, image: np.ndarray) -> Tuple[bool, str]:
        """
        Validate image for acne detection
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check image dimensions
        if len(image.shape) != 3:
            return False, "Image must be in color (3 channels)"
        
        height, width = image.shape[:2]
        
        # Check minimum size
        if height < 100 or width < 100:
            return False, "Image too small (minimum 100x100 pixels)"
        
        # Check maximum size
        if height > 4000 or width > 4000:
            return False, "Image too large (maximum 4000x4000 pixels)"
        
        # Check if image is too dark or too bright
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 30:
            return False, "Image too dark for analysis"
        
        if mean_brightness > 220:
            return False, "Image too bright for analysis"
        
        return True, "Image is valid"
    
    def draw_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: Input image
            detections: List of detection results
            
        Returns:
            Image with drawn detections
        """
        result_image = image.copy()
        
        # Define colors for different acne types
        colors = {
            "acne": (0, 0, 255),            # Red (for general acne detection)
            "blackheads": (0, 0, 255),      # Red
            "whiteheads": (255, 0, 0),      # Blue
            "papules": (0, 255, 0),         # Green
            "pustules": (0, 255, 255),      # Yellow
            "nodules": (255, 0, 255),       # Magenta
            "dark_spots": (128, 0, 128)     # Purple
        }
        
        for detection in detections:
            bbox = detection["bbox"]
            class_name = detection["class"]
            confidence = detection["confidence"]
            
            x1, y1, x2, y2 = map(int, bbox)
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_image
    
    def save_image(self, image: np.ndarray, filepath: str) -> bool:
        """
        Save image to file
        
        Args:
            image: Image to save
            filepath: Output file path
            
        Returns:
            Success status
        """
        try:
            cv2.imwrite(filepath, image)
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
