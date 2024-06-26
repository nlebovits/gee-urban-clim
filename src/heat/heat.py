import argparse

from .utils import make_training_data, predict, train_and_evaluate


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
