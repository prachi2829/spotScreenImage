import os
import glob
import joblib
import numpy as np

from features import extract_features

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

VALID_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")


def get_image_paths(folder):

    paths = []

    for ext in VALID_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, ext)))

    return sorted(paths)


def load_dataset():

    X = []
    y = []

    real_images = get_image_paths("dataset/real")
    screen_images = get_image_paths("dataset/screen")

    print(f"Real images   : {len(real_images)}")
    print(f"Screen images : {len(screen_images)}")

    # Real = 0
    for img in real_images:

        X.append(extract_features(img))
        y.append(0)

    # Screen = 1
    for img in screen_images:

        X.append(extract_features(img))
        y.append(1)

    return np.array(X), np.array(y)


model = Pipeline([

    ("scaler", StandardScaler()),

    ("classifier",
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )

])


if __name__ == "__main__":

    X, y = load_dataset()

    print("\nFeature Vector Shape :", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42

    )


    # Cross Validation

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    print("\nCross Validation Accuracy")

    print(scores)

    print(f"\nMean Accuracy : {scores.mean():.4f}")

    
    # Train
    

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n==============================")

    print("Test Accuracy")

    print("==============================")

    print(accuracy_score(y_test, predictions))

    print("\nClassification Report")

    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix")

    print(confusion_matrix(y_test, predictions))


    # Save Model
    

    os.makedirs("models", exist_ok=True)

    model.fit(X, y)

    joblib.dump(
        model,
        "models/logistic_model.pkl"
    )

    print("\nModel saved to models/logistic_model.pkl")