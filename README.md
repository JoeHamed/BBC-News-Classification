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
### 1. Data Preprocessing
- Text Standardization: Stopwords, punctuation, and lowercasing are removed from the text.
- Text Vectorization: The text data is tokenized and converted into sequences of integers representing words in the vocabulary.
- Label Encoding: The text labels are encoded into integers using the StringLookup layer.
  
### 2. Model Architecture
- Embedding Layer: Converts integer sequences into dense vectors of fixed size.
- Global Average Pooling: Averages the embeddings for each sequence.
- Dense Layer: A fully connected layer with ReLU activation.
- Output Layer: A softmax activation for multi-class classification (`5` categories).
  
### 3. Model Training
- Early Stopping: The training stops once the model reaches `95%` accuracy on the training set and `90%` accuracy on the validation set.
Loss and Accuracy: The model is trained using sparse_categorical_crossentropy loss and accuracy as the evaluation metric.

### 4. Evaluation and Predictions
- The model is evaluated on the test data, and predictions are made for unseen data.

## Usage
### 1. Prepare the Data
Make sure you have the BBC News dataset in CSV format. The dataset should have two columns:
- The first column is the label (category of the news).
- The second column contains the text of the article.
```csv
category,text
business,"The stock market is doing great today..."
entertainment,"The latest movie release has grossed millions..."
...
```
### 2. Run the Script
Run the following command to train the model:
```bash
python main.py
```
### 3. Check the Results
Once the model has trained, it will:
- Plot graphs for training and validation accuracy/loss.
- Save the vocabulary and learned word embeddings into `metadata.tsv` and `weights.tsv`

## Visualizations
The training process includes visualizations of:
- **Accuracy**: How the model's accuracy improves over epochs.
- **Loss**: How the model's loss decreases over time.
These graphs are displayed using `matplotlib` for easy tracking of the model’s performance.
![image](https://github.com/user-attachments/assets/bbdb7005-2f9b-410b-985c-7a9b43470354)

![image](https://github.com/user-attachments/assets/e873bb3b-03e3-49fc-900c-e1cb8e055874)

## Visualizing Word Embeddings
- After training the model, the word embeddings are saved to vecs.tsv (embedding vectors) and meta.tsv (words). These files can be used for visualizing the word vectors in 2D using tools like t-SNE or TensorBoard.
- You can also visualize it using: https://projector.tensorflow.org/

![image](https://github.com/user-attachments/assets/3a6d221b-293e-4a1a-bbd1-486f7ea03dcb)


## Model Files
After training, the following files will be saved:
- `metadata.tsv`: Vocabulary used by the model.
- `weights.tsv`: The learned word embeddings.
These files can be used for further analysis or to inspect the trained model's performance.

