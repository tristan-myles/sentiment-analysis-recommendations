import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.utilities.inter_rater_stats import (
    cohens_kappa,
    confusion_matrix,
    fleiss_kappa,
    neutral_confusion,
    polar_confusion,
    total_confusion,
    update_stats_json,
)

DATA_DIR = Path(__file__).resolve().parents[1]
RAW_UK_DIR = DATA_DIR / "inputs" / "annotation_samples" / "multiple_annotations" / "UK"
OUTPUT_XLSX = DATA_DIR / "inputs" / "annotation_samples" / "final_annotated_samples" / "annotated_sampled_dataset_UK.xlsx"
STATS_FILE = DATA_DIR / "outputs" / "interannotator_agreement.json"

SENTIMENT_LABELS = [-1, 0, 1]

BAD_ROW_TWEET_ID = "1249620065261170688"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine the three UK annotator exports (annotator 1, 2, "
                     "3) into the final annotated UK dataset")
    parser.add_argument(
        "-i", "--input_folder", type=str, default=str(RAW_UK_DIR), metavar="\b",
        help="Folder containing annotated_sampled_dataset_UK_Annotator{1,2,3}.xlsx")
    parser.add_argument(
        "-o", "--output_file", type=str, default=str(OUTPUT_XLSX), metavar="\b",
        help="Where to write the combined dataset (feeds create-folds)")
    return parser.parse_args()


def _load_annotator(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [c.lower().replace("-", ".") for c in df.columns]
    df["tweet_id"] = df["tweet_id"].astype(str)
    df["not.english"] = df["not.english"].fillna(0)
    return df


def _match(df: pd.DataFrame, col_a: str, col_b: str) -> pd.Series:
    a, b = df[col_a], df[col_b]
    return pd.Series(
        np.where(a.isna() | b.isna(), np.nan, np.where(a == b, 1.0, 0.0)),
        index=df.index,
    )


def main(args: argparse.Namespace) -> None:
    input_folder = Path(args.input_folder)
    output_file = Path(args.output_file)

    annotator_1 = _load_annotator(input_folder / "annotated_sampled_dataset_UK_Annotator1.xlsx")

    annotator_2 = _load_annotator(input_folder / "annotated_sampled_dataset_UK_Annotator2.xlsx")
    assert len(annotator_2.columns) == len(annotator_1.columns), (
        "Annotator 2's annotation file has a different number of columns to annotator 1's. "
        "The original R script blindly renames annotator 2's columns to match annotator 1's "
        "by position, so a mismatch here would silently mislabel columns.")
    annotator_2.columns = annotator_1.columns

    annotator_2.loc[annotator_2["tweet_id"] == BAD_ROW_TWEET_ID , "uk.related"] = 0

    annotator_3 = _load_annotator(input_folder / "annotated_sampled_dataset_UK_Annotator3.xlsx")
    assert len(annotator_3.columns) == len(annotator_1.columns), (
        "Annotator 3's annotation file has a different number of columns to annotator 1's.")
    annotator_3.columns = annotator_1.columns
    # R: colnames(rhys_uk)[5:10] (1-indexed) -> pandas columns[4:10] (0-indexed)
    annotator_3 = annotator_3.rename(columns={c: f"{c}_annotator_3" for c in annotator_3.columns[4:10]})

    join_cols = ["tweet_id", "created_at", "text", "search_term"]
    combined = annotator_1.merge(annotator_2, how="left", on=join_cols, suffixes=("_annotator_1", "_annotator_2"))
    combined = combined.merge(annotator_3, how="left", on=join_cols)

    for prefix, col in [
        ("covid_match", "covid.related"),
        ("uk_match", "uk.related"),
        ("sentiment_match", "sentiment"),
        ("not_english_match", "not.english"),
    ]:
        combined[f"{prefix}_a2a3"] = _match(combined, f"{col}_annotator_2", f"{col}_annotator_3")
        combined[f"{prefix}_a1a2"] = _match(combined, f"{col}_annotator_2", f"{col}_annotator_1")
        combined[f"{prefix}_a1a3"] = _match(combined, f"{col}_annotator_1", f"{col}_annotator_3")
        combined[f"{prefix}_sum"] = combined[[f"{prefix}_a2a3", f"{prefix}_a1a2", f"{prefix}_a1a3"]].sum(
            axis=1, skipna=False)

    # three tweets all three annotators agreed were not in English
    n_not_english_agreed = int(
        ((combined["not.english_annotator_1"] == -1) & (combined["not_english_match_sum"] == 3)).sum())

    filtered = combined[combined["not.english_annotator_1"] != -1]
    filtered = filtered[filtered["sentiment_annotator_2"].notna()]

    pct_complete_agreement = float((filtered["sentiment_match_sum"] == 3).mean())

    fleiss = fleiss_kappa(
        filtered[["sentiment_annotator_1", "sentiment_annotator_2", "sentiment_annotator_3"]], SENTIMENT_LABELS)

    pairwise_cohens = {
        "annotator_1_annotator_2": cohens_kappa(
            filtered["sentiment_annotator_1"], filtered["sentiment_annotator_2"], SENTIMENT_LABELS),
        "annotator_1_annotator_3": cohens_kappa(
            filtered["sentiment_annotator_1"], filtered["sentiment_annotator_3"], SENTIMENT_LABELS),
        "annotator_2_annotator_3": cohens_kappa(
            filtered["sentiment_annotator_2"], filtered["sentiment_annotator_3"], SENTIMENT_LABELS),
    }

    sentiment_matched = filtered[filtered["sentiment_match_sum"] > 0]

    final = sentiment_matched[
        sentiment_matched["uk_match_sum"].notna() & (sentiment_matched["uk_match_sum"] != 0)
        & sentiment_matched["covid_match_sum"].notna() & (sentiment_matched["covid_match_sum"] != 0)
    ]

    final = final.copy()
    final["Sentiment"] = np.where(
        final["sentiment_match_a2a3"] == 1, final["sentiment_annotator_2"],
        np.where(final["sentiment_match_a1a2"] == 1, final["sentiment_annotator_2"], final["sentiment_annotator_1"]))
    final["COVID-related"] = np.where(
        final["covid_match_a2a3"] == 1, final["covid.related_annotator_2"],
        np.where(final["covid_match_a1a2"] == 1, final["covid.related_annotator_2"], final["covid.related_annotator_1"]))
    final["UK-related"] = np.where(
        final["uk_match_a2a3"] == 1, final["uk.related_annotator_2"],
        np.where(final["uk_match_a1a2"] == 1, final["uk.related_annotator_2"], final["uk.related_annotator_1"]))
    final["Not-English"] = final["not.english_annotator_1"]

    output_columns = ["tweet_id", "created_at", "text", "search_term",
                       "Sentiment", "COVID-related", "UK-related", "Not-English"]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final[output_columns].to_excel(output_file, index=False)
    print(f"Wrote {len(final)} rows to {output_file}")

    n_no_majority_sentiment = len(filtered) - len(sentiment_matched)
    n_no_majority_covid_or_uk = len(sentiment_matched) - len(final)

    stats = {
        "n_after_language_filter": len(filtered),
        "pct_complete_agreement": round(pct_complete_agreement, 4),
        "fleiss_kappa_overall": round(fleiss["overall"], 4),
        "fleiss_kappa_negative": round(fleiss["per_category"][-1], 4),
        "fleiss_kappa_neutral": round(fleiss["per_category"][0], 4),
        "fleiss_kappa_positive": round(fleiss["per_category"][1], 4),
        "pairwise_cohens_kappa": {k: round(v, 4) for k, v in pairwise_cohens.items()},
        "n_after_sentiment_match_filter": len(sentiment_matched),
        "n_final": len(final),
        "pct_majority_labeled": round(len(sentiment_matched) / len(filtered), 4),
        "n_no_majority_sentiment": n_no_majority_sentiment,
        "n_no_majority_covid_or_uk": n_no_majority_covid_or_uk,
        "n_excluded_total": n_no_majority_sentiment + n_no_majority_covid_or_uk,
        "n_not_english_agreed": n_not_english_agreed,
    }

    confusion = {}
    pairs = [("annotator_1", "annotator_2"), ("annotator_1", "annotator_3"), ("annotator_2", "annotator_3")]
    total_neutral = total_misclass = 0
    for a, b in pairs:
        cm = confusion_matrix(filtered[f"sentiment_{a}"], filtered[f"sentiment_{b}"], SENTIMENT_LABELS)
        nc, pc, tc = neutral_confusion(cm), polar_confusion(cm), total_confusion(cm)
        confusion[f"{a}_vs_{b}"] = {"neutral_confusion": nc, "polar_confusion": pc, "total_misclassifications": tc}
        total_neutral += nc
        total_misclass += tc
    confusion["aggregate"] = {
        "neutral_confusion": total_neutral,
        "total_misclassifications": total_misclass,
        "pct_neutral_of_misclassifications": round(100 * total_neutral / total_misclass, 1),
    }

    update_stats_json(STATS_FILE, "uk", {"stats": stats, "confusion": confusion})

    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    ARGS = parse_arguments()
    main(ARGS)
