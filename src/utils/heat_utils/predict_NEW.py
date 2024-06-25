import os
import ee
import argparse
from dotenv import load_dotenv
from google.cloud import storage
from collections import Counter

from src.config.config import HEAT_MODEL_ASSET_ID, HEAT_SCALE, HEAT_INPUT_PROPERTIES
from src.utils.general_utils.data_exists import data_exists
from src.utils.general_utils.monitor_ee_tasks import monitor_tasks, start_export_task
from src.utils.general_utils.pygeoboundaries import get_adm_ee

load_dotenv()
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_BUCKET = os.getenv("GOOGLE_CLOUD_BUCKET")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

ee.Initialize(project=GOOGLE_CLOUD_PROJECT)


def get_area_of_interest(place_name):
    """Retrieve the area of interest based on the place name."""
    return get_adm_ee(territories=place_name, adm="ADM0").geometry().bounds()


def process_data_to_classify(bbox):
    """Prepare the data to be classified."""
    landcover = ee.Image("ESA/WorldCover/v100/2020").select("Map").clip(bbox)
    dem = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM").mosaic().clip(bbox)

    image_to_classify = (
        landcover.rename("landcover")
        .addBands(dem.rename("elevation"))
        .addBands(ee.Image.pixelLonLat())
    )

    return image_to_classify


def classify_image(image_to_classify, input_properties, model_asset_id):
    """Classify the image using the pre-trained model."""
    regressor = ee.Classifier.load(model_asset_id)
    return image_to_classify.select(input_properties).classify(regressor)


def initialize_storage_client(project, bucket_name):
    """Initialize the Google Cloud Storage client."""
    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(bucket_name)
    return bucket


def export_predictions(classified_image, place_name, bucket, directory_name, scale):
    """Export the predictions to Google Cloud Storage."""
    snake_case_place_name = place_name.replace(" ", "_").lower()
    predicted_image_filename = f"predicted_median_top5_{snake_case_place_name}"

    # Ensure the directory exists by uploading an empty file
    blob = bucket.blob(directory_name)
    blob.upload_from_string(
        "", content_type="application/x-www-form-urlencoded;charset=UTF-8"
    )

    # Start export task
    task = start_export_task(
        classified_image,
        f"{place_name} predicted median temp of top 5 hottest observations",
        bucket.name,
        directory_name + predicted_image_filename,
        scale,
    )
    return task


def main(place_name):
    """Main function to predict heat for a given place and export the result."""
    print("Processing data to classify...")
    snake_case_place_name = place_name.replace(" ", "_").lower()
    directory_name = f"data/{snake_case_place_name}/outputs/"
    bbox = get_area_of_interest(place_name)
    image_to_classify = process_data_to_classify(bbox)
    classified_image = classify_image(
        image_to_classify, HEAT_INPUT_PROPERTIES, HEAT_MODEL_ASSET_ID
    )

    # Initialize the storage client and export predictions
    bucket = initialize_storage_client(GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_BUCKET)
    task = export_predictions(
        classified_image, place_name, bucket, directory_name, HEAT_SCALE
    )

    # Monitor the export task
    monitor_tasks([task], 600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict heat for a given place and export the result."
    )
    parser.add_argument(
        "place_name", type=str, help="Name of the place for which to predict heat"
    )

    args = parser.parse_args()
    main(args.place_name)
