from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytz
import requests
from shapely.geometry import MultiPoint, Point

TIME = f"{datetime(2023, 12, 1, 17)}"


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


def local_to_utc(
    local_datetime, local_timezone="Europe/London", timestamp=True
):
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
    df = pd.concat(
        [weekly_df.drop(["components"], axis=1), expanded_df], axis=1
    )
    df = df.drop(columns=["main"])
    return df


def get_data_openweathermap(path):
    path = Path(INPUTS, "sample_locations.geojson")
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
    return np.expand_dims(df[df["datetime"] == TIME]["pm2_5"].values, axis=1)


# if __name__ == "__main__":
#     extend_train_data(np.array([[51.52728104, -0.24752401]]), ["co"])
