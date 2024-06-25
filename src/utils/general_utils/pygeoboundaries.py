from typing import List, Union

import ee
import geojson
import requests
from fuzzywuzzy import process
from requests_cache import CachedSession

from src.utils.general_utils import countries_iso_dict, iso_codes


class SessionManager:
    def __init__(self):
        self._session = None

    def get_session(self):
        if self._session is None:
            self._session = CachedSession(expire_after=604800)  # Default to 1 week
        return self._session

    def clear_cache(self):
        if self._session:
            self._session.cache.clear()

    def set_cache_expire_time(self, seconds: int):
        self._session = CachedSession(expire_after=seconds)

    def disable_cache(self):
        self._session = requests.Session()


# Instantiate SessionManager
session_manager = SessionManager()


def _is_valid_adm(iso3, adm: str) -> bool:
    session = session_manager.get_session()
    html = session.get(
        f"https://www.geoboundaries.org/api/current/gbOpen/{iso3}/", verify=True
    ).text
    return adm in html


def _validate_adm(adm: Union[str, int]) -> str:
    if isinstance(adm, int) or len(str(adm)) == 1:
        adm = "ADM" + str(adm)
    if str.upper(adm) in [f"ADM{i}" for i in range(6)] or str.upper(adm) == "ALL":
        return str.upper(adm)
    raise KeyError("Invalid ADM level provided.")


def _get_smallest_adm(iso3):
    current_adm = 5
    while current_adm >= 0:
        if _is_valid_adm(iso3, f"ADM{current_adm}"):
            break
        current_adm -= 1
    print(f"Smallest ADM level found for {iso3} : ADM{current_adm}")
    return f"ADM{current_adm}"


def _is_valid_iso3_code(territory: str) -> bool:
    return str.lower(territory) in iso_codes.iso_codes


def _get_iso3_from_name_or_iso2(name: str) -> str:
    name_lower = str.lower(name)

    # Try to get a direct match first
    if name_lower in countries_iso_dict.countries_iso3:
        return str.upper(countries_iso_dict.countries_iso3[name_lower])

    # If no direct match, use fuzzy matching to find the closest key
    closest_match, match_score = process.extractOne(
        name_lower, countries_iso_dict.countries_iso3.keys()
    )

    # Set a threshold for the match score to consider it a valid match
    # You might need to adjust this based on your testing
    if match_score >= 80:  # Assuming a threshold of 80%
        return str.upper(countries_iso_dict.countries_iso3[closest_match])

    # If no match found, log the issue and raise an exception
    print(f"Failed to find a close match for '{name}'")
    raise KeyError(f"Couldn't find country named '{name}'")


def _generate_url(territory: str, adm: Union[str, int]) -> str:
    iso3 = (
        str.upper(territory)
        if _is_valid_iso3_code(territory)
        else _get_iso3_from_name_or_iso2(territory)
    )
    if adm != -1:
        adm = _validate_adm(adm)
    else:
        adm = _get_smallest_adm(iso3)
    if not _is_valid_adm(iso3, adm):
        raise KeyError(
            f"ADM level '{adm}' doesn't exist for country '{territory}' ({iso3})"
        )
    return f"https://www.geoboundaries.org/api/current/gbOpen/{iso3}/{adm}/"


def get_metadata(territory: str, adm: Union[str, int]) -> dict:
    session = session_manager.get_session()
    url = _generate_url(territory, adm)
    response = session.get(url, verify=True)
    response.raise_for_status()  # Raises error for bad responses
    return response.json()


def _get_data(territory: str, adm: str, simplified: bool) -> dict:
    geom_complexity = "simplifiedGeometryGeoJSON" if simplified else "gjDownloadURL"
    try:
        json_uri = get_metadata(territory, adm)[geom_complexity]
        session = session_manager.get_session()
        response = session.get(json_uri)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(
            f"Error while requesting geoboundaries API\nURL: {json_uri}\nException: {e}"
        )
        raise


def get_adm(
    territories: Union[str, List[str]], adm: Union[str, int], simplified=True
) -> dict:
    if isinstance(territories, str):
        territories = [territories]
    geojson_features = [
        geojson.loads(_get_data(i, adm, simplified)) for i in territories
    ]
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            feature["features"][0] for feature in geojson_features
        ],  # Assuming each call returns a FeatureCollection with one feature
    }
    return feature_collection


def get_adm_ee(
    territories: Union[str, List[str]], adm: Union[str, int], simplified=True
):
    # Use the original get_adm function to get the GeoJSON FeatureCollection
    geojson_feature_collection = get_adm(territories, adm, simplified)

    # Convert the GeoJSON FeatureCollection dict to an Earth Engine FeatureCollection
    ee_feature_collection = ee.FeatureCollection(geojson_feature_collection["features"])

    return ee_feature_collection
