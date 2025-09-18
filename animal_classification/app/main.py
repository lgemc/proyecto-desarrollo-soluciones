from pathlib import Path
from contextlib import asynccontextmanager
import sys

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from animal_classification.inference.onnx_classifier import ONNXInference
from animal_classification.utils.logger import Logger

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = Logger(__name__)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Starting Animal Classification API")
    classifier.setup()
    logger.info("Classifier setup completed")
    yield
    logger.info("Shutting down Animal Classification API")
    classifier.teardown()
    logger.info("Classifier teardown completed")

app = FastAPI(title="Animal Classification API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = ONNXInference(model_path=project_root / 'models' / 'animal-classifier-resnet.onnx')

# Mount static files for frontend
static_path = project_root / 'static'
if static_path.exists():
    app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/api/v1/health")
async def health_check():
    logger.debug("Health check requested")
    return {"message": "ok"}

@app.post("/api/v1/classify")
async def classify_image(image: UploadFile = File(...)):
    logger.info(f"Classification request for file: {image.filename}")
    
    if not image.content_type.startswith('image/'):
        logger.warning(f"Invalid file type received: {image.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_bytes = await image.read()
        logger.debug(f"Processing image of size: {len(image_bytes)} bytes")
        result = classifier.predict_with_confidence_from_bytes(image_bytes)
        logger.info(f"Classification successful: {result}")
        return result
    except Exception as e:
        logger.error(f"Classification failed for {image.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# Serve frontend at root path
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """Serve frontend files for all non-API routes"""
    # Don't serve frontend for API routes
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")

    static_path = project_root / 'static'
    if not static_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not available")

    # Try to serve the requested file
    file_path = static_path / full_path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)

    # For SPA, serve index.html for all other routes
    index_path = static_path / 'index.html'
    if index_path.exists():
        return FileResponse(index_path)

    raise HTTPException(status_code=404, detail="Not found")