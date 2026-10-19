
import argparse
import joblib
import pandas as pd
from pathlib import Path

# Locate the saved model relative to this script
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "california_housing_model.joblib"

# Load the trained model
model = joblib.load(MODEL_PATH)

# Features required by the model, in training order
FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def predict_housing(input_data):
    """Validate housing features and predict district-level median values."""
    missing = [feature for feature in FEATURES if feature not in input_data.columns]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    input_data = input_data[FEATURES]

    if input_data.isnull().any().any():
        raise ValueError("Input data contains missing values.")

    predictions = model.predict(input_data)

    return predictions * 100_000


def main():
    parser = argparse.ArgumentParser(
        description="Predict California district-level median house values."
    )

    parser.add_argument(
        "--file",
        help="Path to a CSV file containing housing features."
    )

    args = parser.parse_args()

    if args.file:
        input_data = pd.read_csv(args.file)
    else:
        print("Enter the housing district's features:")

        values = {}

        for feature in FEATURES:
            values[feature] = float(input(f"{feature}: "))

        input_data = pd.DataFrame([values])

    predictions = predict_housing(input_data)

    results = input_data.copy()
    results["PredictedMedianHouseValue_USD"] = predictions

    if args.file:
        output_path = Path(args.file).with_name(
            f"{Path(args.file).stem}_predictions.csv"
        )
        results.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")
    else:
        print(
            f"\nPredicted district-level median house value: "
            f"${predictions[0]:,.2f}"
        )


if __name__ == "__main__":
    main()
