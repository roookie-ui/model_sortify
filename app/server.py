from fastapi import FastAPI, UploadFile, File
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import shutil
import os
import gdown

# === KONFIGURASI GOOGLE DRIVE ===
# Ganti ID ini dengan ID file .h5 milikmu dari Google Drive
GDRIVE_FILE_ID = "13gtfNnyhF1Nq8cX_JRvQBXFUaB5crIHu"  # <-- Ganti dengan ID kamu
MODEL_PATH = "app/model_klasifikasi_sampah.keras"

# === CEK & DOWNLOAD MODEL ===
if not os.path.exists(MODEL_PATH):
    print("Model belum ada, mendownload dari Google Drive...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)



# === LOAD MODEL ===
model = load_model(MODEL_PATH)

# === LABEL KELAS ===
class_names = np.array(['metal', 'battery', 'plastic', 'shoes', 'paper', 'cardboard', 'glass', 'biological'])

# === INISIALISASI API ===
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Sampah Classifier API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Simpan file upload sementara
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Preprocessing gambar
        img = image.load_img(temp_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        prediction = model.predict(img_array)
        pred_class = class_names[np.argmax(prediction)]

        return {
            "predicted_class": pred_class,
            "confidence": float(np.max(prediction))
        }
    finally:
        os.remove(temp_path)
