from flask import Flask, request, jsonify
from model.model import load_model

app = Flask(__name__)
model = load_model()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    http_request = data.get("text")

    # TODO: xử lý request NLP → vector → model.predict

    prediction = model.predict([http_request])[0]
    result = "Attack" if prediction == 1 else "Normal"

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
