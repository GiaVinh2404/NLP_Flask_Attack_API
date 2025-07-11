import os
from flask import Flask, request, jsonify, render_template
from datetime import datetime
from model.model import load_model
from transformers import RobertaTokenizer
import torch

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model()
model.to(device)
model.eval()

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# In-memory log for visualization
logs = []

@app.route("/")
def home():
    return "NLP Attack Detection API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        http_request = data.get("text")

        if not http_request:
            return jsonify({"error": "Missing 'text' in request"}), 400

        # Encode và đưa về device
        inputs = tokenizer(http_request, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Dự đoán
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        result = "Attack" if prediction == 1 else "Normal"

        # Lưu log để trực quan hóa
        logs.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "label": result
        })

        return jsonify({"result": result})
    
    except Exception as e:
        print("[ERROR]", str(e))
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500


@app.route("/data", methods=["GET"])
def get_data():
    """Return recent classification logs for inspection."""
    last_logs = logs[-20:]
    return jsonify({
        "timestamps": [entry["timestamp"] for entry in last_logs],
        "results": [entry["label"] for entry in last_logs]
    })


@app.route("/stats", methods=["GET"])
def get_stats():
    """Return cumulative attack count for plotting."""
    last_logs = logs[-20:]
    timestamps = [entry["timestamp"] for entry in last_logs]
    counts = []
    attack_count = 0
    for entry in last_logs:
        if entry["label"] == "Attack":
            attack_count += 1
        counts.append(attack_count)
    return jsonify({
        "timestamps": timestamps,
        "counts": counts
    })

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True,host="0.0.0.0", port=port)
