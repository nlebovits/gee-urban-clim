import logging
import os
from pathlib import Path


HEAT_MODEL_ASSET_ID = "projects/gee-urban-clim/assets/heat-model-01"
"""The storage location of the current heat model asset in GEE. Update the number at the end of the asset ID as you iterate through models."""

FLOOD_MODEL_ASSET_ID = "projects/gee-urban-clim/assets/flood-model-00"
"""The storage location of the current flood model asset in GEE. Update the number at the end of the asset ID as you iterate through models."""

TRAINING_DATA_COUNTRIES = ["Costa Rica", "Netherlands", "Ghana"]
"""The countries that will be used to train the models."""

EMDAT_DATA_PATH = "emdat/public_emdat_2024-06-24.xlsx"
"""The path in GCS where you have stored the downloaded EMDAT data. These are used to identify known flood dates for the flood module."""


def is_docker() -> bool:
    """
    Whether we are running in Docker or not, e.g. in IDE or CLI environment.
    """
    cgroup = Path("/proc/self/cgroup")
    return (
        Path("/.dockerenv").is_file()
        or cgroup.is_file()
        and "docker" in cgroup.read_text(encoding="utf-8")
    )
