<a href="https://colab.research.google.com/github/xKDR/order_substantive_classification/blob/main/whether_substantive.ipynb" target="_parent">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" width="200"/>
</a>



# Court Order Classification: Substantive vs Non-Substantive Hearings

This repository contains code and resources for automating the classification of court orders into **substantive** and **non-substantive** categories using machine learning (ML) models and large language models (LLMs). The project leverages text extraction techniques, natural language processing (NLP), and model-based classification to streamline the analysis of legal documents.

## Overview

Court hearings produce numerous orders daily, but not all contribute to resolving a case. Distinguishing between substantive hearings (those advancing case resolution) and non-substantive ones (e.g., adjournments) is critical for analyzing court performance. Manual classification is labor-intensive, so this project automates the process by applying data science techniques to classify hearings based on their textual content.

We have implemented two primary approaches:
- **Machine Learning Classifier (LightGBM)**: Achieves 89% accuracy with labeled training data.
- **Large Language Model (LLM) Classifier**: Achieves 81% accuracy without needing labeled training data, reducing the cost of data preparation.

All code is designed to run on **Google Colab**, ensuring reproducibility and accessibility.

## Features

- **Text Extraction**: Extracts text from court order PDFs using `pdfplumber` and `pytesseract` (OCR).
- **Preprocessing**: Tokenizes, cleans, and normalizes text for further analysis.
- **Machine Learning Models**: Trains classifiers such as LightGBM and Naive Bayes on extracted text.
- **Large Language Model (LLM) Integration**: Leverages the pre-trained capabilities of LLMs via **Ollama** for unsupervised classification.
- **Binary Classification**: Predicts whether each court order is substantive or non-substantive.

## Run the project in **Google Colab**:
   - All notebooks are available and pre-configured for Colab execution.
   - Simply upload the notebook to Colab or access the shared link to start running the models.

## Usage

1. **Text Extraction**: Extract text from court order PDFs using the `pdfplumber` library or OCR with Tesseract.
   
2. **Preprocessing**: Clean and preprocess the text using tokenization, lemmatization, and stopword removal.

3. **Classification**:
   - Train the machine learning classifier (LightGBM) using labeled data.
   - Alternatively, use the LLM-based classifier to categorize court orders without the need for labeled training data.

4. **Results**:
   - Machine Learning Classifier: 89% accuracy.
   - Large Language Model (LLM) Classifier: 81% accuracy.

## Data

The dataset used consists of court orders from various hearings. Each order is classified into two categories:
- **Substantive**: Court orders advancing the resolution of a case.
- **Non-Substantive**: Orders that are procedural or involve adjournments.

For reproducibility, we use an SQLite database to store the court orders, which can be processed and classified via the scripts provided.

## Methodology

1. **Extract PDFs**: The orders are stored in an SQLite database as BLOBs. The script retrieves the BLOB data and converts it into text.
2. **Preprocess Text**: Text data is cleaned and transformed using tokenization, TF-IDF vectorization, and lemmatization.
3. **Model Training**: 
   - **LightGBM Classifier**: Trains using supervised learning on the extracted features.
   - **LLM Classifier**: Uses pre-trained language models to classify without labeled training data.
4. **Evaluation**: Evaluate model performance using accuracy, precision, recall, and F1-score.

## Contributing

Contributions are welcome! If you find a bug or have a feature request, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
