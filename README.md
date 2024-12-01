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
```

## Dataset
The dataset used in this project is the BBC News dataset, which contains news articles classified into five categories:
- **business**
- **entertainment**
- **politics**
- **sport**
- **tech**

The data is available in CSV format with two columns:
- **Category** (the label)
- **Text** (the article content)

## File Structure
- `bbc-text.csv`: The dataset containing the text and categories.
- `metadata.tsv`: A file that contains the vocabulary used in the model.
- `weights.tsv`: A file containing the learned word embeddings (excluding the OOV and padding tokens).

## Code Overview
1. Data Preprocessing
Text Standardization: Stopwords, punctuation, and lowercasing are removed from the text.
Text Vectorization: The text data is tokenized and converted into sequences of integers representing words in the vocabulary.
Label Encoding: The text labels are encoded into integers using the StringLookup layer.
2. Model Architecture
Embedding Layer: Converts integer sequences into dense vectors of fixed size.
Global Average Pooling: Averages the embeddings for each sequence.
Dense Layer: A fully connected layer with ReLU activation.
Output Layer: A softmax activation for multi-class classification (5 categories).
3. Model Training
Early Stopping: The training stops once the model reaches 95% accuracy on the training set and 90% accuracy on the validation set.
Loss and Accuracy: The model is trained using sparse_categorical_crossentropy loss and accuracy as the evaluation metric.
4. Evaluation and Predictions
The model is evaluated on the test data, and predictions are made for unseen data.
