# BBC News Archive

import io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
VOCAB_SIZE = 10000
EMBEDDING_DIM = 16
MAX_LENGTH = 120
BATCH_SIZE = 32
TRAINING_SPLIT = 0.8


class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs['accuracy'] >= 0.95 and logs['val_accuracy'] >= 0.90:
            self.model.stop_training = True
            print(f'Reached 95% accuracy and 90% val-accuracy '
                  f'after {epoch} epochs')


def train_val_dataset(texts, labels):
    '''Splits the data into training and validation sets'''

    # Computing the num of sentences that will be used fot training
    train_size = int(TRAINING_SPLIT * len(texts))

    # Splitting into train/validation sets
    train_texts = texts[:train_size]
    validation_texts = texts[train_size:]
    train_labels = labels[:train_size]
    validation_labels = labels[train_size:]

    # Create the train and validation datasets from the splits
    train_dataset = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_texts, validation_labels))

    return train_dataset, validation_dataset

def standralize_func(sentence):
    ''' Removing StopWords, Removing punctuation and Making all the words lowercase'''
    # List of the StopWords
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
                 "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did",
                 "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
                 "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
                 "into", "is", "it", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on",
                 "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
                 "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
                 "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
                 "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "why",
                 "with", "would", "you", "your", "yours", "yourself", "yourselves", "'m", "'d", "'ll", "'re", "'ve",
                 "'s", "'d"]
    # Making all the words lower case
    sentence = tf.strings.lower(sentence)

    # Removing the stop words
    for word in stopwords:
        if word[0] == "'":
            sentence = tf.strings.regex_replace(sentence, fr"{word}\b", "")
        else:
            sentence = tf.strings.regex_replace(sentence, fr'\b{word}\b', "")
    # Remove punctuation
    sentence = tf.strings.regex_replace(sentence, r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']', "")

    return sentence

def fit_vectorizer(train_sequences, standralize_func):
    '''Defines and Adapts the text tokenizer'''

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_sequence_length=MAX_LENGTH,
        standardize=standralize_func,
    )

    vectorizer.adapt(train_sequences)
    return vectorizer

def fit_label_encoder(train_labels_only, validation_labels_only):
    ''' Encoding the categories (text labels) - Encoded as integers'''
    # merging both sets together
    labels = train_labels_only.concatenate(validation_labels_only)

    # Instantiating the StringLookUp layer with no OOV tokens (maps strings to "possibly encoded" indices)
    label_encoder = tf.keras.layers.StringLookup(
        num_oov_indices	=0,)

    # Fit the TextVectorization on the train_labels
    label_encoder.adapt(labels)
    return label_encoder

def preprocess_dataset(dataset, text_vectorizer, label_encoder):
    '''Applying the preprocessing to the dataset'''
    # Converting the dataset sentences to sequences
    dataset = dataset.map(lambda text, label: (text_vectorizer(text), label_encoder(label)))
    dataset = dataset.batch(BATCH_SIZE) # Set the batch size to 32
    return dataset

def create_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(MAX_LENGTH,)),
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=5, activation="softmax"),
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    # 'sparse_categorical_crossentropy' is suitable when the labels are integer.
    # The 'categorical_crossentropy' is used when the labels are 'one-hot encoded'
    return model

def plot_graphs(history, metric):
    plt.plot(history.history[metric]) # Accuracy / Loss
    plt.plot(history.history[f'val_{metric}']) # Val Accuracy / Loss
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, f'val_{metric}'])
    plt.show()

with open('data/bbc-text.csv', 'r', encoding='utf-8') as f:
    print('Header : ', f.readline())
    print('First data point : ', f.readline())

# Loading the data from the csv file
dataset = pd.read_csv('data/bbc-text.csv')
print(dataset.to_numpy().shape)
print(f'The 1st sentence have {len(dataset.to_numpy()[0,1].split())} words')
print(f"The first 5 labels are : {dataset.to_numpy()[:5,0]}") # :5 rows

# Splitting the texts and labels
texts = dataset.iloc[:,-1].values
labels = dataset.iloc[:,0].values
print(texts.shape, type(texts))
print(labels.shape, type(labels))

# Create the datasets (train, validation)
train_dataset , validation_dataset = train_val_dataset(texts, labels)
print(f'There are {train_dataset.cardinality()} training points '
      f'and {validation_dataset.cardinality()} validation points ')

test_sentence = "Hello! We're just about to see this function in action =)"
# Standardizing the sentence
standralized_sentence = standralize_func(test_sentence)
print(f'The sentence is : {test_sentence}, after Standardizing : {standralized_sentence}')

# Creating the vectorizer
text_only_dataset = train_dataset.map(lambda text, label : text)
vectorizer = fit_vectorizer(text_only_dataset, standralize_func)
vocab_size = vectorizer.vocabulary_size()
print(f'The vocabulary size is : {vocab_size}')

# Creating the label encoder
train_labels_only = train_dataset.map(lambda text, label : label)
validation_labels_only = validation_dataset.map(lambda text, label : label)
label_encoder = fit_label_encoder(train_labels_only, validation_labels_only)
print(f'unique labels : {label_encoder.get_vocabulary()}')

# Preprocessing the dataset
train_proc_dataset = preprocess_dataset(dataset= train_dataset, text_vectorizer=vectorizer , label_encoder=label_encoder )
validation_proc_dataset = preprocess_dataset(dataset= validation_dataset, text_vectorizer=vectorizer , label_encoder=label_encoder )
print(f'Number of batches in the train dataset : {train_proc_dataset.cardinality()}')
print(f'Number of batches in the validation dataset : {validation_proc_dataset.cardinality()}')

train_batch = next(train_proc_dataset.as_numpy_iterator())
validation_batch = next(validation_proc_dataset.as_numpy_iterator())
print(f'Train batch : {train_batch}, \n Shape : {train_batch[0].shape} ')
print(f'Validation batch : {validation_batch}, \n Shape : {validation_batch[0].shape}')

# Creating an Untrained model
model = create_model()
# Fitting the model
callback = EarlyStoppingCallback()
history = model.fit(train_proc_dataset, validation_data=validation_proc_dataset, epochs=30, callbacks=[callback])

# Checking if the model is compatible with the dataset
example_batch = train_proc_dataset.take(1)
try:
    model.evaluate(example_batch, verbose=0)
except:
    print('Your Model is Not Compatible with the dataset')
else:
    predictions = model.predict(example_batch, verbose=0)
    print(f'Predictions have the shape : {predictions.shape}')

plot_graphs(history,'accuracy')
plot_graphs(history, 'loss')


## Open writable files
embedding = model.layers[0]

with open('./metadata.tsv', "w") as f:
    for word in vectorizer.get_vocabulary():
        f.write("{}\n".format(word))

weights = tf.Variable(embedding.get_weights()[0][1:]) # excluding the first row (OOV or padding)
# tf.Variable = mutable tensor
with open('./weights.tsv', 'w') as f:
    for w in weights:
        f.write('\t'.join([str(x) for x in w.numpy()]) + "\n")