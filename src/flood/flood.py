import argparse

from .utils import generate_and_export_training_data as make_training_data
from .utils import predict
from .utils import process_all_flood_data as train_and_evaluate


def main(place_name: str) -> None:
    """
    Run the flood prediction pipeline for a given place.

    Args:
        place_name (str): The name of the place to generate flood hazard predictions for.
    """
    make_training_data()
    train_and_evaluate()

    print(f"Predicting for {place_name}...")
    predict(place_name)
    print(f"Prediction for {place_name} completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run flood prediction pipeline for a given place."
    )
    parser.add_argument(
        "place_name", type=str, help="The name of the place to predict on"
    )

    args = parser.parse_args()
    main(args.place_name)
