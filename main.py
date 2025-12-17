from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import uuid
from pathlib import Path
import traceback
from database import init_db, save_new_image, get_image_path, save_detection, get_detection_stats
from classifier import FractureClassifier
from detector import FractureDetector

app = FastAPI(
    title="Fracture Detection System",
    description="X-ray analysis for fracture diagnosis",
    version="3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
MODELS_DIR = os.path.join(BASE_DIR, "data", "models")
INDEX_HTML_PATH = os.path.join(BASE_DIR, "index.html")

CLASSIFIER_PATH = os.path.join(MODELS_DIR, "best_model.pt")
DETECTOR_PATH = os.path.join(MODELS_DIR, "detector.pth")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

init_db()

classifier = None
detector = None
device = 'cpu'

if os.path.exists(CLASSIFIER_PATH):
    classifier = FractureClassifier(CLASSIFIER_PATH, device=device)
    print("Classifier loaded")
else:
    print(f"Warning: Classifier not found at {CLASSIFIER_PATH}")

if os.path.exists(DETECTOR_PATH):
    detector = FractureDetector(DETECTOR_PATH, device=device)
    print("Detector loaded")
else:
    print(f"Warning: Detector not found at {DETECTOR_PATH}")

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.get("/")
async def root():
    if os.path.exists(INDEX_HTML_PATH):
        return FileResponse(INDEX_HTML_PATH)
    return {"error": "index.html not found. Please put it next to main.py"}


@app.post("/api/v1/upload")
async def upload(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be image")
    
    ext = Path(file.filename).suffix
    filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    contents = await file.read()
    with open(filepath, "wb") as f:
        f.write(contents)
    
    image_id = save_new_image(filename, filepath, len(contents))
    print(f"Uploaded successfully: ID={image_id}")
    
    return {
        "message": "Image uploaded",
        "image_id": image_id,
        "filename": filename}

@app.post("/api/v1/predict/{image_id}")
async def predict(image_id: int):
    if not classifier:
        raise HTTPException(status_code=500, detail="Classifier not loaded")
    
    filepath = get_image_path(image_id)
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image file not found in DB/Disk")
    
    class_name, confidence = classifier.predict(filepath)
    filename = os.path.basename(filepath)
    visual_url = None
    bbox_coords = None
    
    if detector:
        bbox_filename = f"bbox_{filename}"
        bbox_path = os.path.join(UPLOAD_DIR, bbox_filename)
        det_result = detector.predict(filepath, save_path=bbox_path)

        if os.path.exists(bbox_path):
            visual_url = f"/uploads/{bbox_filename}"
            bbox_coords = det_result.get("box")

    save_detection(image_id, filename, class_name, confidence, bbox=bbox_coords)
    
    return {
        "message": "Classification completed",
        "image_id": image_id,
        "result": {"class": class_name,"probability": round(confidence, 4),"bbox_url": visual_url}}

@app.get("/api/v1/stats")
async def stats():
    return get_detection_stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
