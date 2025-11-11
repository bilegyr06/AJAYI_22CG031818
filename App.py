from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sqlite3
from PIL import Image
import io
from datetime import datetime

app = FastAPI(title="CSC415 Emotion Detection Agent")
templates = Jinja2Templates(directory="templates")
model = load_model('face_emotionModel.h5')

# DB Setup
conn = sqlite3.connect('database.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS emotions 
             (id INTEGER PRIMARY KEY, user_email TEXT, emotion TEXT, timestamp TEXT, image_path TEXT)''')
conn.commit()

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_emotion(request: Request, file: UploadFile = File(...), email: str = Form(...)):  # FIXED: Request first (no default; FastAPI injects it)
    contents = await file.read()
    
    # NEW: Face Detection with OpenCV (bonus for partial observability, Lecture 2a)
    # Convert to OpenCV format (RGB to BGR for cascade)
    pil_img = Image.open(io.BytesIO(contents)).convert('RGB')
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Load pre-trained Haar cascade (no training needed – expert system, Lecture 1b)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces on grayscale (efficiency trade-off, outcome iv)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        # Crop largest face (utility-based: max area for best prediction)
        areas = [w * h for (_, _, w, h) in faces]
        largest_idx = np.argmax(areas)
        (x, y, w, h) = faces[largest_idx]
        face_crop = cv_img[y:y+h, x:x+w]
        # Convert back to PIL grayscale for model input
        img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)).convert('L')
        print(f"Face detected and cropped: {w}x{h} at ({x}, {y})")  # Log for debugging
    else:
        # Fallback: Use full image (handles no-face cases, adaptability Lecture 2b)
        img = Image.open(io.BytesIO(contents)).convert('L')
        print("No face detected – using full image")
    
    # Existing preprocessing & prediction (now on cropped/grayscale face)
    img_array = np.array(img.resize((48, 48))) / 255.0
    img_array = img_array.reshape(1, 48, 48, 1)
    pred = model.predict(img_array, verbose=0)
    emotion_idx = np.argmax(pred)
    emotion = emotions[emotion_idx]
    confidence = float(pred[0][emotion_idx])

    # Store in DB (unchanged – sequential action, Lecture 2a)
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO emotions (user_email, emotion, timestamp, image_path) VALUES (?, ?, ?, ?)",
              (email, emotion, timestamp, file.filename))
    conn.commit()

    # Render template dynamically with vars (instead of JSON return)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "emotion": emotion,
        "confidence": confidence
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)