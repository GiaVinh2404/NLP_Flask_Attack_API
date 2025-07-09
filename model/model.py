import pickle
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Load model
with open("MODEL_PATH", "rb") as f:
    model = pickle.load(f)

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Dự đoán 1 HTTP request
def predict_request(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        return "Attack" if predicted_class == 1 else "Normal"
