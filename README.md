# BBC News Text Classification

This project demonstrates a text classification pipeline using the BBC News dataset. The goal is to classify news articles into five categories based on their content. The code uses TensorFlow and Keras to train a machine learning model that processes text data and makes predictions.

---

## Features

- **Text Preprocessing**: Removal of stopwords, punctuation, and text normalization (making words lowercase).
- **Model Architecture**: A deep learning model based on embedding layers, followed by a dense network for classification.
- **Dataset**: BBC News dataset, which contains articles categorized into five topics.
- **Early Stopping**: Model training stops early if certain accuracy thresholds are met to avoid overfitting.
- **Visualization**: Plots training and validation accuracy and loss over time.

---

## Hardware Requirements

- A computer or cloud environment capable of running TensorFlow.

## Software Requirements

- Python 3.x
- TensorFlow 2.x
- Pandas
- Matplotlib

You can install the required dependencies via `pip`:

```bash
pip install tensorflow pandas matplotlib numpy
