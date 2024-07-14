import argparse

from flood.flood import main as flood_main
from heat.heat import main as heat_main


def process_place(place_name):
    print(f"Running flood prediction for {place_name}...")
    flood_main(place_name)
    print(f"Flood prediction for {place_name} completed.")

    print(f"Running heat prediction for {place_name}...")
    heat_main(place_name)
    print(f"Heat prediction for {place_name} completed.")


def main(place_names):
    for place_name in place_names:
        process_place(place_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run climate hazard prediction pipelines for given places."
    )
    parser.add_argument(
        "place_names", type=str, nargs="+", help="The names of the places to predict on"
    )

    args = parser.parse_args()
    main(args.place_names)
