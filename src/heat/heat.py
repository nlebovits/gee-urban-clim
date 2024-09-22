import argparse
import os

from google.cloud.storage import Bucket

from src.config.config import HEAT_MODEL_ASSET_ID
from src.constants.constants import HEAT_INPUT_PROPERTIES, HEAT_OUTPUTS_PATH, HEAT_SCALE
from src.heat.utils import make_training_data, train_and_evaluate
from src.utils.utils import (
    initialize_storage_client,
    make_snake_case,
    predict,
)

from .utils import process_data_to_classify


def main(place_name: str) -> None:
    """
    Run the heat prediction pipeline for a given place.

    Args:
        place_name (str): The name of the place to generate heat hazard predictions for.
    """
    make_training_data()
    train_and_evaluate()

    print(f"Predicting for {place_name}...")

    # Preparing arguments for the generic predict function
    GOOGLE_CLOUD_PROJECT: str | None = os.getenv("GOOGLE_CLOUD_PROJECT")
    GOOGLE_CLOUD_BUCKET: str | None = os.getenv("GOOGLE_CLOUD_BUCKET")

    if GOOGLE_CLOUD_PROJECT is None or GOOGLE_CLOUD_BUCKET is None:
        raise ValueError(
            "Google Cloud project or bucket environment variable is not set."
        )

    bucket: Bucket = initialize_storage_client(
        GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_BUCKET
    )

    snake_case_place_name: str = make_snake_case(place_name)

    predict(
        place_name,
        f"predicted_heat_hazard_{snake_case_place_name}",
        bucket,
        HEAT_OUTPUTS_PATH,
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
