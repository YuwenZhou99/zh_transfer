## Overview

This repository contains Python scripts for evaluating models on SLING and CLiMP and using the 'Inseq' library for integrated gradients. The models can be specified through command-line arguments.

## Requirements

Ensure you have Python >= 3.9, <= 3.12. The required Python packages are listed in `requirements.txt`.

## Installation

  1. Clone the repository:
  
  ```bash
  git clone https://github.com/YuwenZhou99/zh_transfer.git
  ```

  2. Install the required packages:

  ```bash
  pip install -r requirements.txt
  ```

## Evaluate models on Chinese benchmarks

`causal_model_CLiMP.py` `masked_model_CLiMP.py` `causal_model_SLING.py` and `masked_model_SLING.py`: These four scripts run a causal or masked model on CLiMP and SLING perspectively, and get the accuracy on the specifc paradigm.

### Example

  ```bash
  python code/masked_model_CLiMP.py --model_name bert-base-chinese
  ```

## Evaluate models with Inseq

`inseq_analsis.py`: This script get the integrated gradients attribution for a specifc minimal pair.

### Usage

  ```bash
  python code/inseq_analysis.py
  --model_name <MODEL_NAME>
  --input_texts <INPUT_TEXT>
  --generated_texts <GENERATED_TEXTS>
  ```

### Arguments

- `--model_name`: Name of the language model to use.
- `--input_texts`: Input texts for the model. Provide as space-separated strings, e.g., "text1" "text2".
- `--generated_texts`: Generated texts for attribution. Provide as space-separated strings, e.g., "generated1" "generated2".

### Example
  ```bash
  python code/masked_model.py
  --model_name 01-ai/Yi-6B
  --input_texts "女歌手离开了"
  --generated_texts "女歌手离开了她。" "女歌手离开了他。"
  ```

## Datasets

The scripts expect the dataset to be in the data directory. Ensure that the dataset files are placed correctly.
