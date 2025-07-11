import os
import requests
import pickle

# Local path to save the model
MODEL_PATH = "api/model/codebert_attack_model.pkl"

# Hugging Face URL to download the model
HUGGINGFACE_URL = "https://huggingface.co/Vinh2404/codebert-attack-model/resolve/main/codebert_attack_model.pkl"

def load_model():
    # If not exists, download the model from Hugging Face
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading model from Hugging Face...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        response = requests.get(HUGGINGFACE_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("[INFO] Download complete.")

    # Load model from local path
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        print("[INFO] Model loaded successfully.")
        print("[INFO] Model type:", type(model))
        return model
