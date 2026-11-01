# BERT and RoBERTa for Automatic Ticket Classification

This repository compares two Transformer-based approaches for assigning customer complaint text to product categories. Both implementations use CFPB complaint data prepared by the companion [ticket classification project](https://github.com/gizemnurbektas/Automatic-ticket-classification-RNN-LSTM).

## Models

| Path | Implementation |
| --- | --- |
| `Bert` | TensorFlow/Keras fine-tuning with `bert-base-uncased`, tokenization, stratified train/validation/test splits, balanced class weights, early stopping, checkpointing, and a classification report. |
| `Roberta` | PyTorch fine-tuning with `roberta-base`, Hugging Face tokenization, weighted loss, a custom training loop, validation-based early stopping, and macro-F1 reporting. |

These are Colab-exported source files without `.py` extensions. They include Colab-specific data upload and notebook commands; open the source in Google Colab and execute it as notebook cells.

## Dataset

The code expects `filtered_complaints.json` with complaint text in `complaint_what_happened` and category labels in `product`. This processed file is not included in the repository; provide it in the Colab runtime before running either implementation.

A related source is [Consumer Complaints Database on Kaggle](https://www.kaggle.com/datasets/sebastienverpile/consumercomplaintsdata), which is listed as **CC0 Public Domain** and sourced from the CFPB. Its original `Consumer_Complaints.csv` is about 413.5 MB and uses the column names `Consumer complaint narrative` and `Product`; it needs filtering and field mapping to create the JSON structure expected here. The full source CSV is not bundled in this repository.

## Run in Google Colab

1. Open `Bert` or `Roberta` in Colab.
2. Install the packages required by that implementation.
3. Provide a processed `filtered_complaints.json` file and run the cells from top to bottom.
4. Review the printed test classification report and training plots.

A GPU is recommended for fine-tuning. Runtime, package versions, available memory, class distribution, and data split can affect execution and results.

## Evaluation

The implementations report class-level precision, recall, and F1. The RoBERTa workflow also tracks macro F1 during validation. Use the outputs from your own run; this repository does not claim a fixed score.

## Author

Gizem Bektaş
