"""
FastAPI main application for AI Acne Identification System
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from pathlib import Path

from ..services.acne_detector import AcneDetector
from ..services.simple_treatment_recommender import SimpleTreatmentRecommender
from ..utils.image_processor import ImageProcessor

# Initialize FastAPI app
app = FastAPI(
    title="AI Acne Identification System",
    description="An intelligent system for acne detection and treatment recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Initialize services
acne_detector = AcneDetector()
treatment_recommender = SimpleTreatmentRecommender()
image_processor = ImageProcessor()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application page"""
    return templates.TemplateResponse("index.html", {"request": {}})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "AI Acne Identification System is running"}

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    detection_mode: str = Form("standard"),
    confidence_threshold: float = Form(0.3)
):
    """
    Upload and analyze facial image for acne detection with multiple modes
    
    Args:
        file: Uploaded image file
        detection_mode: Detection mode ("standard", "sensitive", "aggressive", "multi")
        confidence_threshold: Confidence threshold (0.1-0.9)
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        processed_image = image_processor.preprocess_image(image_data)
        
        # Apply detection mode
        print(f"🔍 Detection mode: {detection_mode}, confidence: {confidence_threshold}")
        try:
            if detection_mode == "sensitive":
                # Lower confidence for more detections
                confidence_threshold = min(confidence_threshold * 0.6, 0.2)
                enhanced_image = image_processor.enhance_image(processed_image)
                
            elif detection_mode == "aggressive":
                # Multiple enhanced versions
                enhanced_image = image_processor.enhance_image_aggressive(processed_image)
                confidence_threshold = min(confidence_threshold * 0.5, 0.15)
                
            elif detection_mode == "multi":
                # Use multiple image versions
                image_versions = image_processor.create_multiple_versions(processed_image)
                all_detections = []
                
                for version in image_versions:
                    detection_results = acne_detector.detect_acne(version, confidence_threshold)
                    all_detections.extend(detection_results["detections"])
                
                # Remove duplicates and get best detections
                final_detections = acne_detector._remove_duplicates(all_detections)
                detection_results = {
                    "detections": final_detections,
                    "total_detections": len(final_detections),
                    "image_shape": processed_image.shape
                }
                
                # Use the best enhanced version for visualization
                enhanced_image = image_versions[0]
                
            else:  # standard mode
                enhanced_image = image_processor.enhance_image(processed_image)
            
            # Detect acne (skip if multi-mode already processed)
            if detection_mode != "multi":
                print(f"🎯 Running detection with mode: {detection_mode}, confidence: {confidence_threshold}")
                detection_results = acne_detector.detect_acne(enhanced_image, confidence_threshold)
                
        except Exception as e:
            # Fallback to standard mode if any detection mode fails
            print(f"Detection mode {detection_mode} failed, falling back to standard: {e}")
            enhanced_image = image_processor.enhance_image(processed_image)
            detection_results = acne_detector.detect_acne(enhanced_image, 0.3)
            detection_mode = "standard"
        
        # Draw bounding boxes on image and save it
        image_with_boxes = image_processor.draw_detections(processed_image, detection_results.get("detections", []))
        
        # Save the image with bounding boxes
        import uuid
        session_id = str(uuid.uuid4())
        result_image_path = f"results archive/result_{session_id}.jpg"
        os.makedirs("static", exist_ok=True)
        image_processor.save_image(image_with_boxes, result_image_path)
        
        # Get treatment recommendations
        treatment_recommendations = treatment_recommender.get_recommendations(detection_results)
        
        # Combine results
        analysis_result = {
            "detections": detection_results,
            "treatments": treatment_recommendations,
            "severity": acne_detector.assess_severity(detection_results),
            "summary": acne_detector.generate_summary(detection_results),
            "result_image": f"/results archive/result_{session_id}.jpg",
            "detection_mode": detection_mode,
            "confidence_threshold_used": confidence_threshold
        }
        
        return {
            "status": "success",
            "analysis": analysis_result,
            "message": "Image analyzed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/treatments")
async def get_treatments():
    """Get available treatment information"""
    return treatment_recommender.get_all_treatments()

@app.get("/acne-types")
async def get_acne_types():
    """Get information about acne types"""
    return {
        "acne_types": [
            {
                "name": "acne",
                "description": "General acne lesions - various types of inflammatory skin conditions including blackheads, whiteheads, papules, pustules, and cysts",
                "severity": "varies"
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
