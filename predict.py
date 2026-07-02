import sys
import joblib
from features import extract_features

MODEL_PATH = "models/logistic_model.pkl"

model = joblib.load(MODEL_PATH)


def predict(image_path):

    features = extract_features(image_path)

    features = features.reshape(1, -1)

    probability = model.predict_proba(features)[0][1]

    return probability


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python predict.py image.jpg")
        sys.exit()

    score = predict(sys.argv[1])

    print(f"{score:.4f}")