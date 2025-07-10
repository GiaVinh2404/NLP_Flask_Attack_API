import os
import requests
import pickle

# Đường dẫn local để lưu model tải về
MODEL_PATH = "api/model/codebert_attack_model.pkl"

# Đường dẫn model trên Hugging Face
HUGGINGFACE_URL = "https://huggingface.co/Vinh2404/codebert-attack-model/resolve/main/codebert_attack_model.pkl"

def load_model():
    # Tải model nếu chưa có
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading model from Hugging Face...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        response = requests.get(HUGGINGFACE_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("[INFO] Download complete.")

    # Load model từ file
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        print("[INFO] Model loaded successfully.")
        return model
