# Georgian Spellcheck Seq2Seq

This project implements a character-level Sequence-to-Sequence (Seq2Seq) model with an Attention mechanism to correct typos and spelling errors in the Georgian language. It includes a custom data corruption pipeline that simulates realistic keyboard typos based on a standard Georgian keyboard layout.

---

### File Structure

* **models.py**: Contains the PyTorch implementation of the Encoder, Attention, Decoder, and Seq2Seq classes.
* **utils.py**: Contains text processing utilities, Levenshtein distance calculation, and the Georgian keyboard corruption logic.
* **data_and_training.ipynb**: Handles word scraping, synthetic dataset generation (creating corrupted pairs), and the model training loop.
* **inference.ipynb**: Used for loading the trained model and evaluating performance on a test sample.
* **requirements.txt**: List of necessary Python libraries.

---

### How to Run

1.  **Install Dependencies** Ensure you have Python installed, then install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Generate Data and Train** Open `data_and_training.ipynb` in a Jupyter environment. 
    * The script will first check for `georgian_words.txt`. If it does not exist, it will scrape words from GitHub.
    * It will generate corrupted versions of these words to create a training set.
    * The training process will save the best model as `georgian_spellcheck_seq2seq.pt`.

3.  **Run Inference** Open `inference.ipynb` to test the model.
    * This notebook loads the saved checkpoint and the word list.
    * It generates 1,000 random test cases and prints metrics like character accuracy and word accuracy.
    * It displays specific examples of words the model successfully fixed and those it failed to correct.