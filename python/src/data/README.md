# data: annotation processing

Raw annotator exports -> one combined, labelled dataset + a 5-fold
cross-validation (CV) split + the released dataset CSV. Every other
`python/src` action reads its input from here.

- **Models:** none
- **`.env`:** none
- **Move to R:** not moved; the model folders read `outputs/` directly

Run in order, from `python/` with the venv active. `run_pipeline.sh` runs
the first three.

## combine_uk

- **What it does:** merges the 3 UK annotators' exports by majority vote on
  sentiment / COVID-relatedness / UK-relatedness.
- **Inputs:** `inputs/annotation_samples/multiple_annotations/UK/`
- **Outputs:** `inputs/annotation_samples/final_annotated_samples/annotated_sampled_dataset_UK.xlsx`;
  UK's agreement stats in `outputs/interannotator_agreement.json`
- **Run:** `python -m src.data.combine_uk`

## compare_samples

- **What it does:** checks FR/DE's primary annotator against a second
  annotator, for agreement only (doesn't change the modelling data).
- **Inputs:** `inputs/annotation_samples/final_annotated_samples/`,
  `inputs/annotation_samples/multiple_annotations/Additional_DE_FR_annotations/`
- **Outputs:** FR/DE's agreement stats in `outputs/interannotator_agreement.json`
- **Run:** `python -m src.data.compare_samples`

## create_fold_data

- **What it does:** combines all 8 languages into one dataset and splits it
  into a 5-fold CV train/val split.
- **Inputs:** `inputs/annotation_samples/final_annotated_samples/`
- **Outputs:** `outputs/complete_cv_dataset.xlsx` (2,530 rows),
  `outputs/train/train_fold_{0-4}.xlsx`, `outputs/val/val_fold_{0-4}.xlsx`
- **Run:** `python -m src.data.create_fold_data`

## create_shareable_dataset

- **What it does:** builds the single CSV released alongside the paper.
- **Inputs:** `inputs/annotation_samples/final_annotated_samples/`,
  `create_fold_data`'s `outputs/{train,val}/`
- **Outputs:** `dataset/annotated_covid_tweets.csv` (see [`dataset/README.md`](../../../dataset/README.md))
- **Run:** `python -m src.data.create_shareable_dataset` (after `create_fold_data`)

## Already in this repo

- `outputs/` is already populated; the commands above regenerate it.
- Tweet text isn't included. In every file here, the `text` column holds
  the tweet ID instead. To re-run the models, rehydrate the tweets from
  their IDs and put the text back in that column.
