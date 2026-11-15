import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.utilities.inter_rater_stats import (
    cohens_kappa,
    confusion_matrix,
    neutral_confusion,
    polar_confusion,
    total_confusion,
    update_stats_json,
)

DATA_DIR = Path(__file__).resolve().parents[1]
MULTILINGUAL_DIR = DATA_DIR / "inputs" / "annotation_samples" / "final_annotated_samples"
SECOND_ANNOTATOR_DIR = DATA_DIR / "inputs" / "annotation_samples" / "multiple_annotations" / "Additional_DE_FR_annotations"
STATS_FILE = DATA_DIR / "outputs" / "interannotator_agreement.json"

SENTIMENT_LABELS = [-1, 0, 1]

LANGUAGE_CONFIG = {
    "fr": {
        "multilingual_code": "FR",
        "file_2": "FR_2.xlsx",
        "language_col": "not.french",
        "country_col": "france.related",
    },
    "de": {
        "multilingual_code": "DE",
        "file_2": "DE_2.xlsx",
        "language_col": "not.german",
        "country_col": "germany.related",
    },
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a second domain expert's FR/German annotations "
                     "against the primary annotator (inputs/annotation_samples/final_annotated_samples/)")
    parser.add_argument(
        "-l", "--language", choices=["fr", "de", "all"], default="all", metavar="\b",
        help="Which language pair to compare")
    parser.add_argument(
        "-m", "--multilingual_folder", type=str, default=str(MULTILINGUAL_DIR), metavar="\b",
        help="Folder containing the primary annotator's annotated_sampled_dataset_{FR,DE}.xlsx")
    parser.add_argument(
        "-i", "--input_folder", type=str, default=str(SECOND_ANNOTATOR_DIR), metavar="\b",
        help="Folder containing the second annotator's {FR,DE}_2.xlsx files")
    return parser.parse_args()


def _load_primary(multilingual_folder: Path, multilingual_code: str) -> pd.DataFrame:
    df = pd.read_excel(
        multilingual_folder / f"annotated_sampled_dataset_{multilingual_code}.xlsx", engine="openpyxl")
    df.columns = [c.lower().replace("-", ".") for c in df.columns]
    df = df.rename(columns={"tweet_id": "id"})
    df["id"] = df["id"].astype(str)
    return df


def _load_second_annotator(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl", dtype={"ID": str})
    df.columns = [c.lower().replace("-", ".") for c in df.columns]
    df["id"] = df["id"].astype(str)
    return df


def compare_language(language: str, multilingual_folder: Path, input_folder: Path) -> dict:
    config = LANGUAGE_CONFIG[language]
    annot_1 = _load_primary(multilingual_folder, config["multilingual_code"])
    annot_2 = _load_second_annotator(input_folder / config["file_2"])

    for col in [config["language_col"], config["country_col"], "covid.related"]:
        annot_1[col] = annot_1[col].fillna(0)
        annot_2[col] = annot_2[col].fillna(0)

    combined = annot_1.merge(annot_2, how="left", on="id", suffixes=("_1", "_2"))

    combined["covid_match"] = np.where(combined["covid.related_1"] == combined["covid.related_2"], 1, 0)
    combined["country_match"] = np.where(
        combined[f"{config['country_col']}_1"] == combined[f"{config['country_col']}_2"], 1, 0)
    combined["sentiment_match"] = np.where(combined["sentiment_1"] == combined["sentiment_2"], 1, 0)
    combined["language_match"] = np.where(
        combined[f"{config['language_col']}_1"] == combined[f"{config['language_col']}_2"], 1, 0)

    filtered = combined[combined["sentiment_1"].notna() & combined["sentiment_2"].notna()]

    overall_kappa = cohens_kappa(filtered["sentiment_1"], filtered["sentiment_2"], SENTIMENT_LABELS)

    per_class_kappa = {}
    for label in SENTIMENT_LABELS:
        a = (filtered["sentiment_1"] == label).astype(int)
        b = (filtered["sentiment_2"] == label).astype(int)
        per_class_kappa[label] = cohens_kappa(a, b, [0, 1])

    cm = confusion_matrix(filtered["sentiment_1"], filtered["sentiment_2"], SENTIMENT_LABELS)
    nc, pc, tc = neutral_confusion(cm), polar_confusion(cm), total_confusion(cm)

    stats = {
        "n_compared": len(filtered),
        "cohens_kappa_overall": round(overall_kappa, 4),
        "cohens_kappa_negative": round(per_class_kappa[-1], 4),
        "cohens_kappa_neutral": round(per_class_kappa[0], 4),
        "cohens_kappa_positive": round(per_class_kappa[1], 4),
        "n_polar_confusion": pc,
        "n_neutral_confusion": nc,
        "n_total_misclassifications": tc,
        "pct_neutral_of_misclassifications": round(100 * nc / tc, 1),
        "confusion_matrix": {str(a): {str(b): int(cm.loc[a, b]) for b in SENTIMENT_LABELS} for a in SENTIMENT_LABELS},
    }

    return stats


def main(args: argparse.Namespace) -> None:
    languages = ["fr", "de"] if args.language == "all" else [args.language]
    multilingual_folder = Path(args.multilingual_folder)
    input_folder = Path(args.input_folder)

    for language in languages:
        stats = compare_language(language, multilingual_folder, input_folder)
        update_stats_json(STATS_FILE, language, stats)

        print(f"\n{language}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    ARGS = parse_arguments()
    main(ARGS)
