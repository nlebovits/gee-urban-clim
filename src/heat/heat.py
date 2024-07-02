import os
import argparse

from .utils import make_training_data, train_and_evaluate
from src.config.config import HEAT_MODEL_ASSET_ID
from src.constants.constants import HEAT_SCALE, HEAT_OUTPUTS_PATH, HEAT_INPUT_PROPERTIES
from src.utils.utils import (
    initialize_storage_client,
    make_snake_case,
    predict,
)
from .utils import process_data_to_classify


def main(place_name):
    make_training_data()
    train_and_evaluate()

    print(f"Predicting for {place_name}...")

    # Preparing arguments for the generic predict function
    GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
    GOOGLE_CLOUD_BUCKET = os.getenv("GOOGLE_CLOUD_BUCKET")
    bucket = initialize_storage_client(GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_BUCKET)

    snake_case_place_name = make_snake_case(place_name)
    directory_name = f"{HEAT_OUTPUTS_PATH}{snake_case_place_name}/"

    predict(
        place_name,
        f"predicted_heat_hazard_{snake_case_place_name}",
        bucket,
        directory_name,
        HEAT_SCALE,
        process_data_to_classify,
        HEAT_INPUT_PROPERTIES,
        HEAT_MODEL_ASSET_ID,
    )

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
