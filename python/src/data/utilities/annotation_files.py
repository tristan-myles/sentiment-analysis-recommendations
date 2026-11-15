from glob import glob
from pathlib import Path

import pandas as pd

from src.data.utilities.harmonise_columns import COMMENT_COLUMNS, COUNTRY_COLUMNS

DATA_DIR = Path(__file__).resolve().parents[1]
INPUT_FOLDER = DATA_DIR / "inputs" / "annotation_samples" / "final_annotated_samples"

ANNOTATION_COLUMNS = (
    [c for cols in COUNTRY_COLUMNS.values() for c in cols] + COMMENT_COLUMNS
    + ["COVID-related", "Hard-to-annotate", "Country-related", "Target-language"])


def load_annotation_files(input_folder=INPUT_FOLDER) -> pd.DataFrame:
    # Sorted by file name: row order fixes the CV folds
    input_files = sorted(glob(str(Path(input_folder).joinpath("*.xlsx"))))
    print(f"found {len(input_files)} files")

    input_dfs = []
    for input_file in input_files:
        df = pd.read_excel(input_file, index_col=None, engine='openpyxl')
        df["country"] = Path(input_file).stem.rsplit("_", 1)[-1]
        input_dfs.append(df)
    input_df = pd.concat(input_dfs)
    input_df.tweet_id = input_df.tweet_id.astype('str')
    return input_df


def drop_annotation_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Pipeline files keep only what the models and R analysis read
    return df.drop(columns=[c for c in ANNOTATION_COLUMNS if c in df.columns])


def drop_annotation_columns_from_file(input_path, output_path=None) -> None:
    # Raw cell values, so ints, strings and blanks are written back as stored
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path
    if input_path.suffix == ".xlsx":
        df = pd.read_excel(input_path, dtype=object, engine="openpyxl")
    else:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])

    df = drop_annotation_columns(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".xlsx":
        df.to_excel(output_path, index=False)
    else:
        # pandas names a blank header "Unnamed: 0"; restore it
        df = df.rename(columns=lambda c: "" if c.startswith("Unnamed: ") else c)
        df.to_csv(output_path, index=False)
