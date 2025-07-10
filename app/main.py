import os
from flask import Flask, request, jsonify, render_template
from datetime import datetime
from app.model.model import load_model

app = Flask(__name__)
model = load_model()

# In-memory log for visualization
logs = []

@app.route("/")
def home():
    return "NLP Attack Detection API is running."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    http_request = data.get("text")

    if not http_request:
        return jsonify({"error": "Missing 'text' in request"}), 400

    prediction = model.predict([http_request])[0]
    result = "Attack" if prediction == 1 else "Normal"

    # Save to logs
    logs.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "label": result
    })

    return jsonify({"result": result})

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
    app.run(host="0.0.0.0", port=port)
