import os
import gdown
import pickle

MODEL_PATH = "model/codebert_attack_model.pkl"
DRIVE_FILE_ID = "1fQcSNlnnfbITLcXMoAjxbB6hbPde7glI"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

def load_model():
    # Tải model nếu chưa có
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading model from Google Drive...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

    # Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        print("[INFO] Model loaded successfully.")
        return model
