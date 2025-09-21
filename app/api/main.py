"""
FastAPI main application for AI Acne Identification System
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
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
async def upload_image(file: UploadFile = File(...)):
    """
    Upload and analyze facial image for acne detection
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        processed_image = image_processor.preprocess_image(image_data)
        
        # Enhance image for better detection
        enhanced_image = image_processor.enhance_image(processed_image)
        
        # Detect acne
        detection_results = acne_detector.detect_acne(enhanced_image)
        
        # Draw bounding boxes on image and save it
        image_with_boxes = image_processor.draw_detections(processed_image, detection_results.get("detections", []))
        
        # Save the image with bounding boxes
        import uuid
        session_id = str(uuid.uuid4())
        result_image_path = f"static/result_{session_id}.jpg"
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
            "result_image": f"/static/result_{session_id}.jpg"
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
