# Georgian Spellcheck Seq2Seq

This project implements a character level Sequence to Sequence model with an Attention mechanism to correct misspelled Georgian words. It includes a custom data corruption pipeline that simulates realistic keyboard typos based on a standard Georgian keyboard layout.

## File Structure

* **models.py**: Implementation of the Encoder, Attention, Decoder, and Seq2Seq classes using PyTorch.
* **utils.py**: Text processing utilities, Levenshtein distance calculation, and Georgian keyboard corruption logic.
* **data_and_training.ipynb**: Notebook for word scraping, synthetic dataset generation, and the model training loop.
* **inference.ipynb**: Notebook for loading the trained model and demonstrating the correct_word function.
* **requirements.txt**: List of necessary Python libraries.

## Setup and Installation

### 1. Create a Virtual Environment

**Windows:**
```
python -m venv venv
venv\Scripts\activate
```

**macOS or Linux:**
```
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```

## How to Run

### Generate Data and Train
Open `data_and_training.ipynb`. This notebook scrapes Georgian words, generates noisy training pairs, and trains the model. The best model will be saved as `georgian_spellcheck_seq2seq.pt`.

### Run Inference
Open `inference.ipynb`. This notebook implements the `correct_word(word: str, model: Seq2Seq)` function and shows the model correcting random examples.
