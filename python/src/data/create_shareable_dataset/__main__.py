import argparse
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.utilities.annotation_files import INPUT_FOLDER, load_annotation_files
from src.data.utilities.harmonise_columns import harmonise_columns

TWITTER_EPOCH_MS = 1288834974657
DATA_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUTS_DIR = DATA_DIR / "outputs"
OUTPUT_CSV = REPO_ROOT / "dataset" / "annotated_covid_tweets.csv"

COLUMNS = ["tweet_id", "created_at", "country", "search_term", "text",
           "Sentiment"]
LABEL_COLUMNS = ["Sentiment"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the shareable dataset (one CSV) from the annotation "
                    "files and the CV folds; run after create_fold_data")
    parser.add_argument(
        "-i", "--input_folder", type=str, default=str(INPUT_FOLDER), metavar="\b",
        help="The folder with the annotation files")
    parser.add_argument(
        "-f", "--folds_folder", type=str, default=str(OUTPUTS_DIR), metavar="\b",
        help="create_fold_data's output folder")
    parser.add_argument(
        "-o", "--output_file", type=str, default=str(OUTPUT_CSV), metavar="\b",
        help="Where to write the CSV")
    return parser.parse_args()


def label_text(value) -> str:
    # 1.0 -> "1"; blank or "-" (both used for missing) become "0" (not sure)
    if pd.isna(value) or str(value).strip() in ("", "-"):
        return "0"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def id_date(tweet_id: str) -> date:
    # Tweet ID: ms since the Twitter epoch in the top bits
    ms = (int(tweet_id) >> 22) + TWITTER_EPOCH_MS
    return datetime.fromtimestamp(ms / 1000, timezone.utc).date()


def iso_date(created_at: str, tweet_id: str) -> str:
    # UK dates mix DD/MM/YYYY and MM/DD/YYYY; pick the reading the ID confirms
    if "/" not in created_at:
        candidates = [date.fromisoformat(created_at)]
    else:
        a, b, year = map(int, created_at.split("/"))
        candidates = []
        for day, month in ((a, b), (b, a)):
            try:
                candidates.append(date(year, month, day))
            except ValueError:
                pass
    posted = id_date(tweet_id)
    matches = {d for d in candidates if abs((d - posted).days) <= 1}
    if len(matches) != 1:
        raise ValueError(f"{tweet_id}: {created_at} doesn't match posted date {posted}")
    return matches.pop().isoformat()


def main(args):
    df = harmonise_columns(load_annotation_files(args.input_folder))

    # Each tweet's validation fold
    fold = {}
    for i in range(5):
        val_df = pd.read_excel(Path(args.folds_folder) / "val" / f"val_fold_{i}.xlsx",
                               dtype={"tweet_id": str})
        fold.update(dict.fromkeys(val_df.tweet_id, i))
    if set(fold) != set(df.tweet_id):
        raise ValueError("Folds don't cover the annotation files exactly")
    df["created_at"] = [iso_date(c, t) for c, t in zip(df.created_at, df.tweet_id)]

    for c in LABEL_COLUMNS:
        df[c] = df[c].map(label_text)

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].rename(columns={"Sentiment": "sentiment"}).to_csv(
        output_file, index=False, encoding="utf-8")
    print(f"{len(df)} tweets -> {output_file}")


if __name__ == "__main__":
    ARGS = parse_arguments()
    main(ARGS)
