#!/usr/bin/env bash
# Annotation pipeline: UK majority vote, FR/DE agreement, CV folds
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/python"

python -m src.data.combine_uk
python -m src.data.compare_samples
python -m src.data.create_fold_data
python -m src.data.create_shareable_dataset
