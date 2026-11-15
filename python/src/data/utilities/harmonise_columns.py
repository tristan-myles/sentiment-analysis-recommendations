"""Merges the per-country annotation columns into one set of columns.

`<Country>-related` -> `Country-related`, the language column (`Not-German`,
..., `Portuguese`) -> `Target-language`, values copied unchanged. Per the
annotation guidelines the language column is 1 = in the language, -1 = not,
0 = not sure. Comment columns are dropped; every other column is kept in place.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# country -> (own country-related column, own language column)
COUNTRY_COLUMNS = {
    "DE": ("Germany-related", "Not-German"),
    "ES": ("Spain-related", "Not-Spanish"),
    "FR": ("France-related", "Not-French"),
    "IT": ("Italy-related", "Not-Italian"),
    "NL": ("Netherlands-related", "Not-Dutch"),
    "PL": ("Poland-related", "Not-Polish"),
    "PT": ("Portugal-related", "Portuguese"),
    "UK": ("UK-related", "Not-English"),
}
COMMENT_COLUMNS = ["Comments", "IH comment", "IH.comment"]


def _own_country_value(df: pd.DataFrame, position: int) -> np.ndarray:
    # positional arrays: the concatenated index repeats (0-299 per country)
    columns = {country: cols[position] for country, cols in COUNTRY_COLUMNS.items()}
    values = np.full(len(df), np.nan, dtype=object)
    for country, column in columns.items():
        own = (df["country"] == country).to_numpy()
        others = [c for c in columns.values() if c != column and c in df.columns]
        if pd.notna(df[others].to_numpy()[own]).any():
            raise ValueError(f"{country} rows have values in {others}")
        if column in df.columns:
            values[own] = df[column].to_numpy()[own]
    return values


def harmonise_columns(df: pd.DataFrame) -> pd.DataFrame:
    unknown = set(df["country"]) - set(COUNTRY_COLUMNS)
    if unknown:
        raise ValueError(f"No column mapping for countries {sorted(unknown)}")

    related = [cols[0] for cols in COUNTRY_COLUMNS.values()]
    language = [cols[1] for cols in COUNTRY_COLUMNS.values()]
    merged = {"Country-related": _own_country_value(df, 0),
              "Target-language": _own_country_value(df, 1)}

    # Merged columns take the place of the first per-country column of each kind
    columns, placed = {}, set()
    for c in df.columns:
        new = "Country-related" if c in related else \
            "Target-language" if c in language else None
        if new and new not in placed:
            columns[new] = merged[new]
            placed.add(new)
        elif not new and c not in COMMENT_COLUMNS:
            columns[c] = df[c].to_numpy()
    return pd.DataFrame(columns, index=df.index)

