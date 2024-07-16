from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO('yolov8n.pt')

@app.post("/detect_cars/")
async def detect_cars(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.25, ge=0, le=1)
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    results = model(image)
    
    detections = []
    for result in results:
        for box in result.boxes:
            if model.names[int(box.cls)] == 'car' and box.conf.item() >= confidence_threshold:
                detections.append({
                    'confidence': float(box.conf),
                    'box': box.xyxy[0].tolist()
                })
    
    # Draw bounding boxes on the image
    img = results[0].plot()
    
    # Convert the image to base64 for sending to frontend
    buffered = io.BytesIO()
    Image.fromarray(img).save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {
        "cars_detected": len(detections),
        "detections": detections,
        "image_with_boxes": img_str
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
