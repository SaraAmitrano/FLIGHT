#!/usr/bin/env python
# coding: utf-8

# %%

# https://github.com/vdhar1992/ASHRAE_Great_energy_predictorIII_CaseStudy/blob/main/ASHRAE_GreatEnergyPredictorIII.ipynb

from astral import LocationInfo
from astral.sun import sunrise, sunset
import pytz
import pandas as pd
import numpy as np
import holidays
from ashrae.utils import DATA_PATH, timer, load_data  # , reduce_mem_usage

# %%

from pandas.api.types import is_datetime64_any_dtype as is_datetime


def reduce_mem_usage(df, use_float16=False):
    """iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage(index=True, deep=True).sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    # iterating over every column and finding the type of the column
    for col in df.columns:
        if is_datetime(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype):
            continue
        col_type = df[col].dtype

        # If the columns is not object
        if col_type != object:
            # Get the minimum and maximum value
            c_min = df[col].min()
            c_max = df[col].max()
            # If the type is int
            if str(col_type).startswith("int"):
                # If the min max values lies with thin the range of int8 type then assign the type as int8
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                # If the min max values lies with thin the range of int16 type then assign the type as int16
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                # If the min max values lies with thin the range of int32 type then assign the type as int32
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                # If the min max values lies with thin the range of int64 type then assign the type as int64
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # If the min max values lies with thin the range of float16 type then assign the type as float16
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                # If the min max values lies with thin the range of float32 type then assign the type as float32
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage(index=True, deep=True).sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def break_datetime(df):
    ts = df["timestamp"]
    df["hour_x"] = np.float16(np.cos(2 * np.pi * ts.dt.hour / 24))
    df["hour_y"] = np.float16(np.sin(2 * np.pi * ts.dt.hour / 24))

    df["day_x"] = np.float16(np.cos(2 * np.pi * ts.dt.day / ts.dt.days_in_month))
    df["day_y"] = np.float16(np.sin(2 * np.pi * ts.dt.day / ts.dt.days_in_month))

    df["weekday_x"] = np.float16(np.cos(2 * np.pi * ts.dt.weekday / 7))
    df["weekday_y"] = np.float16(np.sin(2 * np.pi * ts.dt.weekday / 7))

    df["dayofyear_x"] = np.float16(
        np.cos(2 * np.pi * ts.dt.dayofyear / (ts.dt.is_leap_year + 365))
    )
    df["dayofyear_y"] = np.float16(
        np.sin(2 * np.pi * ts.dt.dayofyear / (ts.dt.is_leap_year + 365))
    )

    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["dayofyear"] = pd.to_datetime(df["timestamp"].dt.date)
    return df


def percent_missing_val(df):

    percent_missing = (df.isnull().sum() * 100) / len(df)
    missing_value_df = pd.DataFrame(
        {"column_name": df.columns, "percent_missing": percent_missing}
    )
    return missing_value_df


def holiday_filler(df, calendar):
    df["is_holiday"] = [0] * (df.shape[0])
    for country, dates in calendar.items():
        df.loc[df["country"] == country, "is_holiday"] = (
            df.loc[df["country"] == country, "timestamp"]
            .dt.date.astype("str")
            .isin(dates)
        ).astype("int")
    return df


# %%
if __name__ == "__main__":
    output_dir = DATA_PATH / "preprocessed"

    with timer("Loading data"):
        data = load_data("data")
        building = load_data("meta")
        weather = load_data("weather")

    with timer("Timestamp conversion"):
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        weather["timestamp"] = pd.to_datetime(weather["timestamp"])

    with timer("Extract missing weather data"):
        assert (
            weather.index == weather.sort_values(["site_id", "timestamp"]).index
        ).all()
        weather = (
            weather.merge(
                data[["timestamp", "building_id"]]
                .merge(building[["building_id", "site_id"]], on="building_id")[
                    ["timestamp", "site_id"]
                ]
                .drop_duplicates(),
                on=["timestamp", "site_id"],
                how="outer",
            )
            .sort_values(["site_id", "timestamp"])
            .reset_index(drop=True)
        )

    with timer("Fill weather missing values"):
        na_cols = weather.columns[weather.isna().any()]
        weather[na_cols] = weather.groupby("site_id")[na_cols].ffill().bfill()

    with timer("Split wind direction"):
        weather["wind_direction_x"] = weather["wind_direction"] * np.cos(
            2 * np.pi * weather["wind_direction"] / 360
        )
        weather["wind_direction_y"] = weather["wind_direction"] * np.sin(
            2 * np.pi * weather["wind_direction"] / 360
        )

    with timer("Add Relative humidity"):
        weather["relative_humidity"] = (
            np.exp(
                (17.67 * weather["dew_temperature"])
                / (243.5 + weather["dew_temperature"])
            )
        ) / (
            np.exp(
                (17.67 * weather["air_temperature"])
                / (243.5 + weather["air_temperature"])
            )
        )

    with timer("Break datetime"):
        data = break_datetime(data)

    with timer("Square meters"):
        building["square_meters"] = building["square_feet"] * (0.3048**2)

    with timer("scale square meters"):
        building["square_meters_log10"] = np.log10(building["square_meters"])

    with timer("Reduce building data"):
        building = building.drop(
            columns=["square_feet", "square_meters", "year_built", "floor_count"]
        )

    with timer("Add location info"):
        location_df = pd.DataFrame()
        location_df["site_id"] = np.arange(0, 16)
        location_df["city"] = [
            "Orlando",
            "Heathrow",
            "Tempe",
            "Washington",
            "Berkeley",
            "Southampton",
            "Washington",
            "Ottawa",
            "Orlando",
            "Austin",
            "Saltlake",
            "Ottawa",
            "Dublin",
            "Minneapolis",
            "Philadelphia",
            "Rochester",
        ]
        location_df["country"] = [
            "US",
            "UK",
            "US",
            "US",
            "US",
            "UK",
            "US",
            "Canada",
            "US",
            "US",
            "US",
            "Canada",
            "Ireland",
            "US",
            "US",
            "US",
        ]

        location_df["timezone"] = [
            "US/Eastern",
            "Europe/London",
            "US/Arizona",
            "US/Eastern",
            "US/Pacific",
            "Europe/London",
            "US/Eastern",
            "US/Eastern",
            "US/Eastern",
            "US/Central",
            "US/Mountain",
            "US/Eastern",
            "Europe/Dublin",
            "US/Central",
            "US/Eastern",
            "US/Central",
        ]

        location_df["latitude"] = [
            28.5383,
            51.4700,
            33.4272,
            38.8898,
            37.8717,
            50.9097,
            38.8898,
            45.4247,
            28.5383,
            30.2667,
            40.7587,
            45.4247,
            53.3501,
            44.9867,
            39.9526,
            44.0163,
        ]
        location_df["longitude"] = [
            -81.3792,
            -0.4543,
            -111.9399,
            -77.0090,
            -122.2728,
            -1.4043,
            -77.0090,
            -75.6950,
            -81.3792,
            -97.7333,
            -111.8762,
            -75.6950,
            -6.2661,
            -93.2581,
            -75.1652,
            -92.4754,
        ]

        building = building.merge(
            location_df[["site_id", "city", "country"]], on="site_id", how="left"
        )

    with timer("Drop zero meters readings"):
        data = data[data["meter_reading"] != 0]

    with timer("Scale target"):
        data["meter_reading_scaled"] = np.log10(data["meter_reading"])
        data = data.drop(columns="meter_reading")

    with timer("Drop bad buildings"):
        data = data[~data["building_id"].isin([1099, 778])]

    with timer("Merge building and weather"):
        data_df = data.merge(building, on="building_id", how="left").merge(
            weather, on=["site_id", "timestamp"], how="left"
        )

    with timer("Add holidays"):
        UK = []
        US = []
        CA = []
        IRE = []

        for ptr in holidays.UnitedKingdom(years=2016).keys():
            UK.append(str(ptr))
        UK.append("2017-01-01")

        for ptr in holidays.UnitedStates(years=2016).keys():
            US.append(str(ptr))
        US.append("2017-01-01")

        for ptr in holidays.Canada(years=2016).keys():
            CA.append(str(ptr))
        CA.append("2017-01-01")

        for ptr in holidays.Ireland(years=2016).keys():
            IRE.append(str(ptr))
        IRE.append("2017-01-01")

        calendar = {"US": US, "UK": UK, "Canada": CA, "Ireland": IRE}
        data_df = holiday_filler(data_df, calendar)
        data_df = data_df.drop(columns=["city", "country"])

    with timer("Add daylight feature"):
        daylight_df = pd.DataFrame()
        daylight_df["dayofyear"] = pd.date_range(
            start="2016-01-01", end="2017-01-01", freq="D"
        )

        daylight_df = (
            daylight_df.assign(key=1)
            .merge(location_df.assign(key=1), on="key")
            .drop("key", axis=1)
        )

        daylight_df["sunrise"] = np.nan
        daylight_df["sunset"] = np.nan

        for i in range(daylight_df.shape[0]):
            city = daylight_df.loc[i, "city"]
            country = daylight_df.loc[i, "country"]
            latitude = daylight_df.loc[i, "latitude"]
            longitude = daylight_df.loc[i, "longitude"]
            timezone = daylight_df.loc[i, "timezone"]
            date_to_check = daylight_df.loc[i, "dayofyear"].to_pydatetime().date()

            cityloc = LocationInfo(city, country, timezone, latitude, longitude)

            local_tz = pytz.timezone(cityloc.timezone)

            s_sunrise = sunrise(cityloc.observer, date=date_to_check, tzinfo=local_tz)
            s_sunset = sunset(cityloc.observer, date=date_to_check, tzinfo=local_tz)

            daylight_df.loc[i, "sunrise"] = (
                pd.to_datetime(s_sunrise.replace(tzinfo=None))
                - (
                    s_sunrise.replace(
                        tzinfo=None, hour=0, minute=0, second=0, microsecond=0
                    )
                )
            ).total_seconds()
            daylight_df.loc[i, "sunset"] = (
                pd.to_datetime(s_sunset.replace(tzinfo=None))
                - (
                    s_sunrise.replace(
                        tzinfo=None, hour=0, minute=0, second=0, microsecond=0
                    )
                )
            ).total_seconds()

        daylight_df["sunrise_hour"] = (
            pd.to_timedelta(daylight_df["sunrise"], unit="s").dt.total_seconds() / 3600
        ).astype(int)
        daylight_df["sunset_hour"] = (
            pd.to_timedelta(daylight_df["sunset"], unit="s").dt.total_seconds() / 3600
        ).astype(int)

        data_df = data_df.merge(
            daylight_df[["site_id", "dayofyear", "sunrise_hour", "sunset_hour"]],
            on=["site_id", "dayofyear"],
            how="left",
        )
        data_df["isdaytime"] = (
            (data_df["timestamp"].dt.hour >= data_df["sunrise_hour"])
            & (data_df["timestamp"].dt.hour <= data_df["sunset_hour"])
        ).astype(int)
        data_df = data_df.drop(columns=["sunrise_hour", "sunset_hour"])

    with timer("Reduce memory usage"):
        data_df = reduce_mem_usage(data_df, use_float16=True)

    with timer("Remove features"):
        data_df = data_df.drop(
            # columns=["building_id", "dew_temperature", "dayofyear", "day", "month"]
            columns=["dew_temperature", "dayofyear", "day", "month"]

        )

    assert (~data_df.isna().any()).all()

    with timer("Save Data"):
        data_df.to_feather(output_dir / "data_clean.feather")

# %%
