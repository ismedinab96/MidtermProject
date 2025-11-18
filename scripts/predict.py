import sys
import json
import argparse
import pickle

def load_artifacts(model_path, dv_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(dv_path, "rb") as f:
        dv = pickle.load(f)
    return model, dv

def predict_single(model, dv, input_json):
    X = dv.transform([input_json])
    pred = model.predict(X)[0]
    return float(pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="../model/model.pkl")
    parser.add_argument("--dv-path", default="../model/dv.pkl")
    args = parser.parse_args()

    body = sys.stdin.read()
    if not body:
        print("Error! no existe el archivo JSON")
        sys.exit(1)
    input_json = json.loads(body)
    model, dv = load_artifacts(args.model_path, args.dv_path)
    pred = predict_single(model, dv, input_json)
    print(json.dumps({"prediction": pred}))
