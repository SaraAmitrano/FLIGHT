# %%

import pandas as pd
from pathlib import Path
import numpy as np

from utils.mem import reduce_mem_usage


# %%


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

    df["year"] = df["timestamp"].dt.year
    return df


# %%

root_path = Path().resolve()
if root_path.name == "scripts_dev":
    root_path = root_path.parent

datasets_path = root_path / "datasets"

dataset = "GEFCom2012"

dataset_path = datasets_path / dataset


load_folder = dataset_path / "Load"

load_files = sorted(list(load_folder.glob("*.csv")))

assert load_files[2].name == "Load_history.csv"

load_history_file = pd.read_csv(load_files[2], index_col=0)

load_history_file.insert(
    0, "date", pd.to_datetime(load_history_file[["year", "month", "day"]])
)
load_history_file = load_history_file.drop(columns=["year", "month", "day"])

load_history_file = load_history_file.apply(
    lambda col: (
        col.str.replace(",", ".").astype("float") if col.dtype == "O" else col
    ),
    axis=0,
)

# convert the index to a column 'zone_id'
load_history_file = load_history_file.reset_index()

# melt the dataframe
df_melted = load_history_file.melt(
    id_vars=["zone_id", "date"], var_name="hour", value_name="load"
)
df_melted["hour"] = df_melted["hour"].str[1:].astype(int) - 1
df_melted["timestamp"] = pd.to_datetime(
    df_melted["date"].astype(str) + " " + df_melted["hour"].astype(str) + ":00:00"
)
df_melted = df_melted.drop(columns=["date", "hour"])
df_melted = df_melted.set_index(["zone_id", "timestamp"]).sort_index()

df_melted = df_melted["load"].unstack(level=0)

df_melted = df_melted[~df_melted.isna().all(axis=1)]

# %%

out_dataset_path = dataset_path / "Load_preprocessed"

out_dataset_path.mkdir(exist_ok=True, parents=True)

df_melted.to_feather(out_dataset_path / "df_prepro.feather")

del df_melted

# %%

df_melted = pd.read_feather(out_dataset_path / "df_prepro.feather")

df_melted = df_melted.reset_index().melt(
    id_vars=["timestamp"], var_name="zone_id", value_name="load"
)

holidays = pd.read_csv(
    dataset_path / "Load/Holiday_List.csv",
    sep=",",
    names=["name", "2004", "2005", "2006", "2007", "2008"],
)
holidays = holidays[1:]

date_holidays = holidays.melt(
    id_vars=["name"], var_name="year", value_name="date"
).dropna()

# change the Friday, December 31, 2004 date to Friday, December 31 and the year to 2004
date_holidays.loc[10, "date"] = "Friday, December 31"
date_holidays.loc[10, "year"] = "2004"

date_holidays["date"] = pd.to_datetime(
    date_holidays["date"] + " " + date_holidays["year"]
)

df_melted["is_holiday"] = 0
df_melted.loc[
    df_melted["timestamp"].dt.date.isin(date_holidays["date"].dt.date), "is_holiday"
] = 1

df_melted = break_datetime(df_melted)

df_melted["year_diff"] = df_melted["year"] - 2004

# %%

df_melted.loc[df_melted["load"] <= 0, "load"] = 1

# %%

df_melted["load"].plot.hist(bins=100)

# %%

df_melted["load_scaled"] = np.log10(df_melted["load"])
df_melted = df_melted.drop(columns=["load"])

# %%

df_melted = df_melted.sort_values(["timestamp", "zone_id"])

# %%

df_melted = df_melted.drop(columns=["dayofyear", "year"])

# %%
# add time blocks

df_melted["ts_block"] = -1
id_start = 0

for zone_id in df_melted["zone_id"].sort_values().unique():
    df_ts = df_melted.loc[df_melted.zone_id == zone_id, "timestamp"]
    df_ts_diff = (df_ts.diff().dt.total_seconds() / 3600).fillna(1).astype(int)
    df_melted.loc[df_melted.zone_id == zone_id, "ts_block"] = (
        df_ts_diff > 1
    ).cumsum() + id_start
    # print(df_melted.loc[df_melted.zone_id == zone_id, "ts_block"].unique())
    id_start = df_melted.loc[df_melted.zone_id == zone_id, "ts_block"].max() + 1

# %%

df_melted = reduce_mem_usage(df_melted)

# %%

df_melted.to_feather(out_dataset_path / "data_clean.feather")

# %%

df_melted_plot = df_melted.set_index(["zone_id", "timestamp"])

# statistic plot varying the zone_id
import matplotlib.pyplot as plt

cm = plt.get_cmap("tab20")
plt.figure(figsize=(10, 6))
for zone_id in df_melted_plot.index.get_level_values(0).unique():
    df_melted_plot.loc[zone_id, "load_scaled"].plot(
        kind="hist", bins=100, alpha=0.5, color=cm(zone_id)
    )
plt.title("Load Distribution for Each Zone")
plt.xlabel("Load")
plt.ylabel("Frequency")
plt.legend(df_melted_plot.index.get_level_values(0).unique())
plt.show()

# %%

# temperature = pd.read_csv(
#     dataset_path / "Load/temperature_history.csv", sep=",", index_col=0
# )
# temperature.index.name = "station_id"
# temperature = temperature.reset_index()
# temperature = temperature.melt(
#     id_vars=["station_id", "year", "month", "day"],
#     var_name="hour",
#     value_name="temperature",
# )
# temperature["hour"] = temperature["hour"].str.extract(r"(\d+)").astype(int)

# temperature["datetime"] = pd.to_datetime(
#     temperature[["year", "month", "day"]]
# ) + pd.to_timedelta(temperature["hour"] - 1, unit="h")

# temperature = temperature.drop(columns=["year", "month", "day", "hour"])

# # multiple column as index
# temperature = temperature.set_index(["station_id", "datetime"])
# temperature.rename(columns={"station_id": "zone_id"}, inplace=True)
# temperature.rename(
#     columns={f"h{i}": f"{i-1}:00:00" for i in range(1, 25)}, inplace=True
# )
# # riordina le colonne in base al datetime e impostalo come indice
# temperature = temperature.sort_index()

# cm = plt.get_cmap("tab20")
# plt.figure(figsize=(10, 6))
# for st_id in temperature.index.get_level_values(0).unique():
#     temperature.loc[st_id, "temperature"].plot(
#         kind="hist", bins=100, alpha=0.5, color=cm(st_id)
#     )
# plt.title("Temperature for Each Station")
# plt.xlabel("Temperature")
# plt.ylabel("Frequency")
# plt.legend(temperature.index.get_level_values(0).unique())
# plt.show()


# %%
