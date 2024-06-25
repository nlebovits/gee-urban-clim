import os

from dotenv import load_dotenv

load_dotenv()
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")

HEAT_SCALE = 90
HEAT_MODEL_ASSET_ID = "projects/gee-urban-clim/assets/heat-model-00"
HEAT_INPUT_PROPERTIES = ["longitude", "latitude", "landcover", "elevation"]
TRAINING_DATA_COUNTRIES = ["Costa Rica", "Netherlands"]
