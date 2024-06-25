## peusdocode:

# we want a script that's run with pipenv run python heat.py "County Name"

# "County Name" will only be used to define the data that we're *predicting* on; the training dataset will not change

# will also want the option to predict on multiple countries, eventually

# Check if the relevant output data exist. If they do, print a message saying they already exist and exit. if not, proceed.

# this will need to involve cross-reference the

# check if the trained model exists. If it does, skip to the prediction step. If not, proceed.

# check if the training data proceed exist. If it does, skip to the model training step. If not, proceed.

## first, generate training data from a list of countries (start with, say, 2 for testing purposes--can scale as needed)

## next, load those images into an image collection; sample it using stratified sampling (how do we handle the landcover classes? maybe randomly sample across the image collection--will basically be representative if sample is large enough)

## train a model on these data; save the model + results to GCS

## use the model to predict for our given study area; save the results to GCP


# CONSTS to define in the .env file:

# CLOUD_PROJECT
# CLOUD_BUCKET
# path to gcloud API key

# also make sure to include progress printing wherever possible


### -------------

import argparse
import ee
from data_utils.process_heat_data import process_heat_data


def main(countries):
    cloud_project = "hotspotstoplight"
    ee.Initialize(project=cloud_project)

    for place_name in countries:
        print("Processing data for", place_name, "...")
        process_heat_data(place_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process heat data for given countries."
    )
    parser.add_argument(
        "countries",
        metavar="Country",
        type=str,
        nargs="+",
        help="A list of countries to process",
    )

    args = parser.parse_args()

    main(args.countries)
