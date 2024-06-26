import argparse

from flood.flood import main as flood_main
from heat.heat import main as heat_main


def main(place_name):
    print(f"Running flood prediction for {place_name}...")
    flood_main(place_name)
    print(f"Flood prediction for {place_name} completed.")

    print(f"Running heat prediction for {place_name}...")
    heat_main(place_name)
    print(f"Heat prediction for {place_name} completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run climate hazard prediction pipelines for a given place."
    )
    parser.add_argument(
        "place_name", type=str, help="The name of the place to predict on"
    )

    args = parser.parse_args()
    main(args.place_name)
