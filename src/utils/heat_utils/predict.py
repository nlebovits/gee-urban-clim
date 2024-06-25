import os

import ee
from dotenv import load_dotenv
from google.cloud import storage

from src.config.config import HEAT_MODEL_ASSET_ID, HEAT_SCALE, HEAT_INPUT_PROPERTIES, TRAINING_DATA_COUNTRIES
from src.utils.general_utils.data_exists import data_exists
from src.utils.general_utils.monitor_ee_tasks import monitor_tasks, start_export_task
from src.utils.general_utils.pygeoboundaries import get_adm_ee

load_dotenv()
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_BUCKET = os.getenv("GOOGLE_CLOUD_BUCKET")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

training_data_countries = TRAINING_DATA_COUNTRIES
cloud_project = GOOGLE_CLOUD_PROJECT
bucket_name = GOOGLE_CLOUD_BUCKET

ee.Initialize(project=cloud_project)

# function inputs are a place name and nothing else

def predict_heat(place_name): 

snake_case_place_name = place_name.replace(" ", "_").lower()

aoi = get_adm_ee(territories=place_name, adm="ADM0")
bbox = aoi.geometry().bounds()

scale = HEAT_SCALE

directory_name = f"data/{snake_case_place_name}/outputs/"

inputProperties = HEAT_INPUT_PROPERTIES

def process_data_to_classify(bbox):

    landcover = ee.Image("ESA/WorldCover/v100/2020").select("Map").clip(bbox)

    dem = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM").mosaic().clip(bbox)

    image_to_classify = (
        landcover.rename("landcover")
        .addBands(dem.rename("elevation"))
        .addBands(ee.Image.pixelLonLat())
    )

    print("Sampling image band names", image_to_classify.bandNames().getInfo())

    return image_to_classify


# Process data and classify the image
image_to_classify = process_data_to_classify(bbox)

regressor = ee.Classifier.load(HEAT_MODEL_ASSET_ID)

classified_image = image_to_classify.select(inputProperties).classify(regressor)

# Initialize the storage client
storage_client = storage.Client(project=cloud_project)
bucket = storage_client.bucket(bucket_name)

# Create and upload an empty file to initialize the directory (if needed)
blob = bucket.blob(directory_name)
blob.upload_from_string(
    "", content_type="application/x-www-form-urlencoded;charset=UTF-8"
)


# Export the predicted image
# Ensure the filename or path for the predicted image is unique to avoid overwriting
predicted_image_filename = f"predicted_median_top5_{snake_case_place_name}"  # Example filename, ensure it's unique
# The function `start_export_task` should handle the export logic, including setting the correct filename/path
task = start_export_task(
    classified_image,
    f"{place_name} Median temp of top 5 hottest observations",
    bucket_name,
    directory_name + predicted_image_filename,
    scale,
)
tasks = [task]
monitor_tasks(tasks, 600)


def process_data_to_classify(bbox):
    landcover = ee.Image("ESA/WorldCover/v100/2020").select("Map").clip(bbox)
    dem = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM").mosaic().clip(bbox)

    image_to_classify = (
        landcover.rename("landcover")
        .addBands(dem.rename("elevation"))
        .addBands(ee.Image.pixelLonLat())
    )

    print("Sampling image band names", image_to_classify.bandNames().getInfo())

    return image_to_classify
