from pathlib import Path
from contextlib import asynccontextmanager
import sys

from fastapi import FastAPI, UploadFile, File, HTTPException
from animal_classification.inference.resnet_classifier import ResNetInference
from animal_classification.shared.logger import Logger

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

classifier = ResNetInference(model_path=project_root / 'models' / 'animal-classifier-resnet.pth')

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