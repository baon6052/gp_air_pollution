import csv
import os
from datetime import datetime, timedelta
from pathlib import Path
import time
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytz
import requests
from shapely.geometry import MultiPoint, Point

CWD = Path.cwd()
DATA_DIR = Path(CWD, "data")
OUTPUTS_DIR = Path(DATA_DIR, "outputs")
API_KEY = "18ea55dba598f506a699783efcc2b07a"  # "ac83f92d098798d935bb5b75cb378802"  # 84bec9a1ca5c364023b8e490b7fc3547#os.environ["API_KEY"]
INPUTS_DIR = Path(DATA_DIR, "inputs")

TIME = datetime(2023, 12, 1, 17)
CLIMATE_VARS = [
    "temp",
    "feels_like",
    "pressure",
    "humidity",
    "temp_min",
    "temp_max",
    "clouds_all",
    "wind_deg",
    "wind_speed",
]


def convert_multipoint_to_point(multipoint: MultiPoint):
    """
    Converts a MultiPoint geometry to a Point geometry, assuming the MultiPoint contains only one point.
    """
    if isinstance(multipoint, MultiPoint) and len(multipoint.geoms) == 1:
        return Point(multipoint.geoms[0])
    return multipoint


def local_to_utc(local_datetime, local_timezone="Europe/London", timestamp=True):
    """
    Convert a local timezone datetime into a UTC timestamp.

    Args:
    local_datetime_str (str): String representing the local datetime, e.g., '2023-01-01 12:00:00'.
    local_timezone (str): String representing the local timezone, e.g., 'America/New_York'.

    Returns:
    int: UTC timestamp.
    """
    # Parse the local datetime string
    # local_datetime = datetime.strptime(local_datetime_str, '%Y-%m-%d %H:%M:%S')

    # Set the timezone for the local datetime
    local_tz = pytz.timezone(local_timezone)
    local_datetime = local_tz.localize(local_datetime)

    # Convert to UTC
    utc_datetime = local_datetime.astimezone(pytz.utc)

    # Return the UTC timestamp
    if timestamp:
        return int(utc_datetime.timestamp())
    else:
        return utc_datetime


def utc_to_local(
    utc_timestamp, local_timezone="Europe/London", timestamp=True
) -> int | datetime:
    """
    Convert a UTC timestamp to a local time datetime object.

    Args:
    utc_timestamp (int): UTC timestamp.
    local_timezone (str): String representing the local timezone,
    e.g., 'America/New_York'.

    Returns:
    datetime: Datetime object in the local timezone.
    """
    # Create a UTC datetime object from the timestamp
    utc_datetime = datetime.utcfromtimestamp(utc_timestamp)
    utc_datetime = utc_datetime.replace(tzinfo=pytz.utc)

    # Convert to the local timezone
    local_tz = pytz.timezone(local_timezone)
    local_datetime = utc_datetime.astimezone(local_tz)

    if timestamp:
        return int(local_datetime.timestamp())
    else:
        return local_datetime


def get_climate_data(
    coordinates: np.array,
    climate_variables: list[str],
    return_df: bool = False,
) -> pd.DataFrame:
    base_url = "https://history.openweathermap.org/data/2.5/history/city"
    start_timestamp = int(TIME.timestamp())
    end_timestamp = int(TIME.timestamp())

    l = ["latitude", "longitude"]
    l.extend(climate_variables)
    climate_variable_data = {k: [] for k in l}

    for coord in coordinates:
        # print(f"Finished: {list(coordinates).index(coord)}/{len(list(coordinates))}")
        latitude = coord[0]
        longitude = coord[1]
        response = requests.get(
            f"{base_url}?lat={latitude}&lon={longitude}&start={start_timestamp}&end={end_timestamp}&appid={API_KEY}"
        )

        if response.status_code == 200:
            data = response.json()["list"][
                0
            ]  # will only be one because its for a specific hour I think
            climate_variable_data["latitude"].append(latitude)
            climate_variable_data["longitude"].append(longitude)
            for climate_var in climate_variables:
                if climate_var in [
                    "temp",
                    "feels_like",
                    "pressure",
                    "humidity",
                    "temp_min",
                    "temp_max",
                ]:
                    climate_variable_data[climate_var].append(data["main"][climate_var])
                elif "wind" in climate_var:
                    climate_variable_data[climate_var].append(
                        data["wind"][climate_var.split("_")[1]]
                    )
                elif "clouds" in climate_var:
                    climate_variable_data[climate_var].append(
                        data["clouds"][climate_var.split("_")[1]]
                    )  # only accepts all
    # arrays = [
    #     np.array(climate_variable_data[key]) for key in climate_variable_data.keys()
    # ]
    df = pd.DataFrame(climate_variable_data)
    if return_df:
        return df
    else:
        return np.array(df.iloc[:, 2:].values.tolist())


def get_air_pollution_data(coordinates: Point, return_df: bool = False) -> pd.DataFrame:
    base_url = "http://api.openweathermap.org/data/2.5/air_pollution/history"

    start_timestamp = int(TIME.timestamp())
    end_timestamp = int(TIME.timestamp())

    all_data = []
    for coord in coordinates:
        latitude = coord[0]
        longitude = coord[1]

        response = requests.get(
            f"{base_url}?lat={latitude}&lon={longitude}&start={start_timestamp}&end={end_timestamp}&appid={API_KEY}"
        )

        if response.status_code == 200:
            data = {}
            response_data = response.json()["list"][0]
            data["latitude"] = latitude
            data["longitude"] = longitude
            data["timestamp"] = start_timestamp
            data.update(response_data["components"])
            all_data.append(data)
    df = pd.DataFrame(all_data)
    if return_df:
        return df
    else:
        return np.array(df[["pm2_5"]].values.tolist())


def extend_train_data(
    coordinates: np.ndarray, climate_variables: list[str]
) -> np.ndarray:
    if not climate_variables:
        return coordinates

    df = get_climate_data(coordinates, climate_variables)

    train_data = np.column_stack((coordinates, df))

    return train_data


def get_cached_openweather_data(
    num_samples: int | None = None, climate_variables: list[str] = []
) -> np.array:
    cached_data = []
    columns = ["latitude", "longitude"]
    columns.extend(climate_variables)

    with open("data/cached_openweather_data.csv", "r", newline="") as csvfile:
        dict_reader = csv.DictReader(csvfile)
        for row in dict_reader:
            if num_samples:
                if num_samples == 0:
                    break
                num_samples -= 1

            cached_data.append(
                np.array([float(value) for key, value in row.items() if key in columns])
            )
    return np.array(cached_data)


def get_cached_air_pollution_data(
    num_samples: int | None, columns: list[str] = ["pm2_5"]
) -> np.ndarray:
    df = pd.read_csv(f"{DATA_DIR}/cached_air_pollution_data.csv")
    if not num_samples:
        return np.array([])

    return df[:num_samples][columns].values


def setup_cached_climate_data(
    coordinates,
    climate_variables: list[str] = [],
):
    climate_variable_data_df = get_climate_data(
        coordinates, climate_variables=climate_variables, return_df=True
    )

    climate_variable_data_df.to_csv("data/cached_openweather_data.csv", index=False)


def generate_air_pollution_cache(coordinates):
    air_pollution_df = get_air_pollution_data(coordinates, return_df=True)

    air_pollution_df.to_csv(f"{DATA_DIR}/cached_air_pollution_data.csv", index=False)
