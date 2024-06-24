## pseudocode:

## identify, say, 20 flood events in different countries + continents.

    ### criteria: want at least 2 events per year since 2016, want at least 2 countries per continent
    ### will need to think more about how we select these countries for diversity but also reasonable export size (e.g., trying to avoid the US)

## save to GCS (if they already exist, skip)

## pull sample from all of these

## train a model (if the model already exists, skip)

## save the model to GCS (plus model accuracy)

## apply the model to a given country that is passed via initial command
    ## should also have some flexibility insofar as the size of the area that this is passed to

## save prediction results to GCS (if they already exist, skip)

##


### ----------------------------------------------
# so, module #1 is create training data, module #2 is train + eval model, module #3 is actually apply model. these are separate.


import argparse
import ee
from data_utils.process_all_data import process_flood_data


def main(countries):
    cloud_project = "hotspotstoplight"
    ee.Initialize(project=cloud_project)

    for place_name in countries:
        print("Processing data for", place_name, "...")
        process_flood_data(place_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process flood data for given countries."
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