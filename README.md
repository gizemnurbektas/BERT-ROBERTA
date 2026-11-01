# BERT and RoBERTa for Automatic Ticket Classification

This repository compares two Transformer-based approaches for assigning customer complaint text to product categories. Both implementations use CFPB complaint data prepared for the companion [ticket classification project](https://github.com/gizemnurbektas/Automatic-ticket-classification-RNN-LSTM).

## Models

| Path | Implementation |
| --- | --- |
| `Bert` | TensorFlow/Keras fine-tuning with `bert-base-uncased`, tokenization, stratified train/validation/test splits, balanced class weights, early stopping, checkpointing, and a classification report. |
| `Roberta` | PyTorch fine-tuning with `roberta-base`, Hugging Face tokenization, weighted loss, a custom training loop, validation-based early stopping, and macro-F1 reporting. |

These are Colab-exported source files without `.py` extensions. They include Colab-specific data upload and notebook commands; open the source in Google Colab and execute it as notebook cells.

## Dataset

The included `filtered_complaints.json` contains 8,308 CFPB consumer complaint narratives across 18 product categories, with up to 500 examples per category. Each record has `complaint_what_happened` and `product` fields, matching the inputs expected by the code. Email addresses, URLs, phone/long-number patterns, and source `XXXX` redactions are masked in this shared sample.

The sample was derived from the [Consumer Complaints Database on Kaggle](https://www.kaggle.com/datasets/sebastienverpile/consumercomplaintsdata), listed as **CC0 Public Domain** and sourced from the CFPB. The original `Consumer_Complaints.csv` is about 413.5 MB; only a balanced, filtered subset is included here. See the Kaggle page for provenance and source details.

## Run in Google Colab

1. Open `Bert` or `Roberta` in Colab.
2. Install the packages required by that implementation.
3. Download `filtered_complaints.json` from this repository and upload it when the code requests the file.
4. Run the cells from top to bottom and review the printed test classification report and training plots.

A GPU is recommended for fine-tuning. Runtime, package versions, available memory, data split, and class distribution can affect execution and results.

## Evaluation

The implementations report class-level precision, recall, and F1. The RoBERTa workflow also tracks macro F1 during validation. Use the outputs from your own run; this repository does not claim a fixed score.

## Author

Gizem Bektaş
