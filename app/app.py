from flask import Flask, request, jsonify
import pickle, os

APP_PORT = int(os.environ.get("PORT", 9696))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
DV_PATH = os.path.join(BASE_DIR, "model", "dv.pkl")

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(DV_PATH, "rb") as f:
        dv = pickle.load(f)
    return model, dv

app = Flask("midterm-service")
model, dv = load_model()

@app.route("/")
def home():
    return "Midterm ML Zoomcamp Cohort 2025- service running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Request body must be JSON"}), 400
    try:
        X = dv.transform([data])
        pred = model.predict(X)[0]
        return jsonify({"prediction": float(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

