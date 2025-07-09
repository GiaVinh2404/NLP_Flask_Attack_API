from flask import Flask, request, jsonify
from model import predict_request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Cho phép PHP hoặc web khác gọi API

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    http_text = data["text"]
    result = predict_request(http_text)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
