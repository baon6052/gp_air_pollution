from pathlib import Path
from datetime import datetime, timedelta
import csv
import os

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
API_KEY = os.environ["API_KEY"]
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


def get_batch_air_pollutant_levels(coordinates: npt.ArrayLike) -> npt.ArrayLike:
    air_pollutant_levels = []

    for coord in coordinates:
        air_pollutant_level = get_air_pollutant_level(coord)
        air_pollutant_levels.append(air_pollutant_level)

    return np.squeeze(np.array(air_pollutant_levels), axis=1)


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


def get_climate_data(longitude_coordinate: float, latitude_coordinate: float):
    API_KEY = "84bec9a1ca5c364023b8e490b7fc3547"
    base_url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    local_timezone = "Europe/London"

    start_date = datetime(2023, 12, 1)
    end_date = start_date + timedelta(days=1)

    dfs = []

    current_date = start_date
    while current_date < end_date:
        start_timestamp = local_to_utc(current_date)
        end_timestamp = local_to_utc(current_date + timedelta(days=1))

        response = requests.get(
            f"{base_url}?lat={latitude_coordinate}&lon={longitude_coordinate}&start={start_timestamp}&end={end_timestamp}&appid={API_KEY}"
        )


def get_air_pollution_data(geometry: Point) -> pd.DataFrame:
    latitude, longitude = geometry.coords[0]
    API_KEY = "84bec9a1ca5c364023b8e490b7fc3547"
    base_url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    local_timezone = "Europe/London"

    start_date = datetime(2023, 12, 1)
    end_date = start_date + timedelta(days=1)

    dfs = []

    current_date = start_date
    while current_date < end_date:
        start_timestamp = local_to_utc(current_date)
        end_timestamp = local_to_utc(current_date + timedelta(days=1))

        response = requests.get(
            f"{base_url}?lat={latitude}&lon={longitude}&start={start_timestamp}&end={end_timestamp}&appid={API_KEY}"
        )
        if response.status_code == 200:
            daily_data = response.json()["list"]
            daily_df = pd.DataFrame(daily_data)
            daily_df["datetime"] = (
                daily_df["dt"]
                .apply(lambda date: utc_to_local(date, timestamp=False))
                .to_list()
            )
            daily_df["datetime_timestamp"] = (
                daily_df["dt"]
                .apply(lambda date: utc_to_local(date, timestamp=True))
                .to_list()
            )
            daily_df = daily_df.drop(columns=["dt"])
            daily_df["latitude"] = latitude
            daily_df["longitude"] = longitude
            dfs.append(daily_df)

        current_date += timedelta(days=1)

    weekly_df = pd.concat(dfs)
    expanded_df = weekly_df["components"].apply(pd.Series)
    # You can then concatenate these new columns back to your original DataFrame
    df = pd.concat([weekly_df.drop(["components"], axis=1), expanded_df], axis=1)
    df = df.drop(columns=["main"])
    return df


def get_data_openweathermap(path):
    path = Path(INPUTS_DIR, "sample_locations.geojson")
    sample_locations_gdf = gpd.read_file(path)
    sample_locations_gdf["geometry"] = sample_locations_gdf["geometry"].apply(
        convert_multipoint_to_point
    )
    list_of_dfs = sample_locations_gdf.apply(
        lambda row: get_air_pollution_data(row["geometry"]), axis=1
    ).to_list()
    large_df = pd.concat(list_of_dfs, ignore_index=True)


def extend_train_data(
    coordinates: np.ndarray, climate_variables: list[str]
) -> np.ndarray:
    if not climate_variables:
        return coordinates

    train_data = []
    for longitude_coordinate, latitude_coordinate in coordinates:
        point = Point(longitude_coordinate, latitude_coordinate)
        df = get_air_pollution_data(point)
        climate_variable_values = [longitude_coordinate, latitude_coordinate]
        for climate_variable in climate_variables:
            climate_variable_values.append(
                df[df["datetime"] == TIME][climate_variable].values[0]
            )

        train_data.append(climate_variable_values)

    return np.array(train_data)


def get_air_pollutant_level(coords: np.ndarray) -> np.ndarray:
    """
    (latitude, long) -> pollutant level for time
    """
    point = Point(coords[0], coords[1])
    df = get_air_pollution_data(point)
    return np.expand_dims(df[df["datetime"] == str(TIME)]["pm2_5"].values, axis=1)


def get_cached_openweather_data(num_samples: int | None = None, climate_variables: list[str] = []) -> np.array:
    cached_data = []
    columns = ["longitude", "latitude"]
    columns.extend(climate_variables)

    with open("data/cached_openweather_data.csv", "r", newline="") as csvfile:
        dict_reader = csv.DictReader(csvfile)
        for row in dict_reader:
            if num_samples:
                if num_samples == 0:
                    break
                num_samples -= 1

            cached_data.append(np.array([float(value) for key, value in row.items() if key in columns]))
    return np.array(cached_data)


def get_climate_data(
    coordinates: np.array,
    climate_variables: list[str],
) -> pd.DataFrame:
    base_url = "https://history.openweathermap.org/data/2.5/history/city"
    start_timestamp = int(TIME.timestamp())
    end_timestamp = int(TIME.timestamp())

    climate_variable_data = {k: [] for k in climate_variables}

    for coord in coordinates:
        latitude = coord[0]
        longitude = coord[1]
        response = requests.get(
            f"{base_url}?lat={latitude}&lon={longitude}&start={start_timestamp}&end={end_timestamp}&appid={API_KEY}"
        )

        if response.status_code == 200:
            data = response.json()["list"][
                0
            ]  # will only be one because its for a specific hour I think
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
    arrays = [
        np.array(climate_variable_data[key]) for key in climate_variable_data.keys()
    ]
    return np.vstack(arrays).T


def setup_cache_data(
    longitude_bounds: tuple[float, float],
    latitude_bounds: tuple[float, float],
    climate_variables: list[str] = [],
    num_samples: int = 1000,
):
    longitude_linear_space = np.linspace(
        start=longitude_bounds[0], stop=longitude_bounds[1], num=num_samples
    )
    latitude_linear_space = np.linspace(
        start=latitude_bounds[0], stop=latitude_bounds[1], num=num_samples
    )

    meshgrid = np.meshgrid(longitude_linear_space, latitude_linear_space)
    coordinates = np.column_stack(
        [meshgrid_dimension.ravel() for meshgrid_dimension in meshgrid]
    )

    # Call out to openweather MAP API here with list of coordinates and climate variables
    climate_variable_data = get_climate_data(
        coordinates, climate_variables=climate_variables
    )
    # After calling, extend coordinate data with climate variable data from API
    # Then save all to csv below

    with open("data/cached_openweather_data.csv", "w", newline="") as csvfile:
        fieldnames = ["longitude", "latitude"]
        fieldnames.extend(climate_variables)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for (longitude_coordinate, latitude_coordinate), climate_data in zip(
            coordinates, climate_variable_data
        ):
            row_data = {
                "longitude": longitude_coordinate,
                "latitude": latitude_coordinate,
            }

            for i, climate_var in enumerate(climate_variables):
                row_data[climate_var] = climate_data[i]
            writer.writerow(row_data)


if __name__ == "__main__":
    longitude_bounds = (51.41728104, 51.56728104)
    latitude_bounds = (-0.24752401, 0.12247599)

    setup_cache_data(
        longitude_bounds,
        latitude_bounds,
        climate_variables=CLIMATE_VARS,
        num_samples=200,
    )

    get_cached_openweather_data()
