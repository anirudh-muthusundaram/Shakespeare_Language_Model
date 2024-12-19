# Volume Text-based (Shakespearean) Language Model

This project builds a character-level language model trained on William Shakespeare's works sourced from [Project Gutenberg](https://www.gutenberg.org). The goal is to generate Shakespeare-like text and evaluate the model’s performance in terms of loss, perplexity, and accuracy.

## Overview

1. **Data Scraping**:  
   A custom Python scraper downloads Shakespeare’s public domain texts directly from Project Gutenberg. It navigates to the Shakespeare collection page, finds all relevant plain-text links, and downloads them.

2. **Data Cleaning**:  
   Project Gutenberg eBooks contain licensing and boilerplate text. A cleaning step removes this non-literary content by identifying “START” and “END” markers, leaving primarily Shakespeare’s text.

3. **Preprocessing**:  
   - The cleaned texts are concatenated into a single dataset.
   - A character-level vocabulary (`chars`, `char_to_idx`, `idx_to_char`) is constructed.
   - The data is split into training and validation sets, and sequences of fixed `seq_length` are created for next-character prediction tasks.

4. **Model Training**:  
   The model is an LSTM-based character-level language model implemented in PyTorch. It is trained to predict the next character given a preceding sequence. Hyperparameters such as `embedding_dim`, `rnn_units`, and `epochs` can be tuned.

5. **Evaluation**:  
   After training, the model is evaluated on a validation set for:
   - **Loss & Perplexity**: Measures how well the model predicts the next character.
   - **Accuracy**: Checks how often the model’s top prediction matches the actual next character.
   
   You can also generate text to qualitatively assess the model’s Shakespeare-like prose.

## Note

- Make sure you use and experiment with the latest version (e.g., _vX).
- Any and all contributions to this language model are welcome.
- This repository will be updated iteratively to improve the model.
- Run gpu.py to check whether your GPU can be used to run the model.
 
## Features

- **Automated Scraping & Cleaning**: Ensures the dataset is primarily Shakespeare’s works.
- **Character-Level Modeling**: Captures stylistic details including punctuation and spacing.
- **Quantitative & Qualitative Evaluation**: Uses numerical metrics and sample generation to evaluate performance.

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
- Ensure you have Python 3.7+ and packages like requests, beautifulsoup4, numpy, torch, and matplotlib.

3. **Download and Clean the Data**:
   ```bash
   python project_Shakespeare
- This creates a shakespeare_works directory with cleaned text files.

4. **Train the Model:**:
   ```bash
   python modelv3.py
- Adjust hyperparameters as needed.
- Once training completes, you’ll have a trained model saved as .pth.

5. **Evaluate and Generate Text**:
   ```bash
   python evaluatev3.py
- Adjust hyperparameters as needed.
- This command prints validation metrics and generates sample text for inspection.

## Results

- Validation losses typically stabilize around 0.7 to 1.0, corresponding to a perplexity near 2.0–3.0.
- Accuracy often surpasses 90%, indicating strong character-level predictions.
- Generated text resembles Shakespeare’s style, though occasional modern or licensing text may appear if not fully cleaned.

## Future Improvements

- **Further Data Cleaning**: Remove remaining non-literary lines if they persist.
- **Model Architecture**: Experiment with Transformers or larger LSTMs.
- **Hyperparameter Tuning**: Vary sequence length, number of training epochs, and other parameters.