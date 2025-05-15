from fastapi import FastAPI, UploadFile, File, Response
import cv2
import numpy as np

app = FastAPI()

@app.post("/detect_edges/")
async def generate_feature_mask(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Apply Canny Edge Detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # Encode image as PNG
    _, encoded_img = cv2.imencode('.png', edges)
    
    return Response(content=encoded_img.tobytes(), media_type="image/png")