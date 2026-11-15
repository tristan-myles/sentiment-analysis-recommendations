import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.data.utilities.annotation_files import (
    INPUT_FOLDER, drop_annotation_columns, load_annotation_files)

DATA_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = DATA_DIR / "outputs"


def parse_arguments() -> argparse.Namespace:
    """
     Argument parser

    :return: An argparse namespace
    """
    parser = argparse.ArgumentParser("")
    parser.set_defaults(which='base')

    parser.add_argument(
        "-i", "--input_folder", type=str, default=str(INPUT_FOLDER), metavar="\b",
        help="The folder that contains the excel input files to create the "
             "K-fold CV dataset from")

    parser.add_argument(
        "-f", "--folds", type=int, default=5, metavar="\b",
        help="Number of folds")

    args = parser.parse_args()
    return args


def main(args):
    input_df = drop_annotation_columns(load_annotation_files(args.input_folder))
    indices = list(input_df.index)

    OUTPUTS_DIR.mkdir(exist_ok=True)

    kf = KFold(n_splits=args.folds, random_state=3141, shuffle=True)
    splits = kf.split(indices)

    val_folder = OUTPUTS_DIR.joinpath("val")
    val_folder.mkdir(exist_ok=True)
    train_folder = OUTPUTS_DIR.joinpath("train")
    train_folder.mkdir(exist_ok=True)

    for i, (train_index, val_index) in enumerate(tqdm(splits)):
        train_df = input_df.iloc[train_index]
        val_df = input_df.iloc[val_index]

        train_df.to_excel(train_folder.joinpath(f"train_fold_{i}.xlsx"),
                          index=False)
        val_df.to_excel(val_folder.joinpath(f"val_fold_{i}.xlsx"),
                        index=False)

        if i == 0:
            pd.concat([train_df, val_df]).to_excel(
                OUTPUTS_DIR.joinpath("complete_cv_dataset.xlsx"), index=False)


if __name__ == "__main__":
    ARGS = parse_arguments()
    main(ARGS)
