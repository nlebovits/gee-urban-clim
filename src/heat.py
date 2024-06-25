import argparse
from src.utils.heat_utils.make_training_data import make_training_data
from src.utils.heat_utils.train_and_eval import train_and_evaluate
from src.utils.heat_utils.predict import predict


def main(place_name):

    make_training_data()

    train_and_evaluate()

    print(f"Predicting for {place_name}...")
    predict(place_name)
    print(f"Prediction for {place_name} completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run heat prediction pipeline for a given place."
    )
    parser.add_argument(
        "place_name", type=str, help="The name of the place to predict on"
    )

    args = parser.parse_args()
    main(args.place_name)
