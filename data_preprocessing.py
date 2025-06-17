import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def s_bert(data):
    """
    Function to generate sentence embeddings using Sentence-BERT.

    Args:
        data (list): List of sentences to encode.

    Returns:
        list: List of sentence embeddings.
    """
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # Small, fast
    embeddings = model.encode(data, convert_to_tensor=True)
    return embeddings

def s_bert_better(data):
    """
    Function to generate sentence embeddings using a more powerful Sentence-BERT model.

    Args:
        data (list): List of sentences to encode.

    Returns:
        list: List of sentence embeddings.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')  # More powerful, slower
    embeddings = model.encode(data, convert_to_tensor=True)
    return embeddings

def s_bert_best(data):
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(data, convert_to_tensor=True)
    return embeddings

def bag_of_words(data, max_features=300):
    """
    Function to generate bag-of-words embeddings from text data using only the most frequent words.

    Args:
        data (list or array-like): List of sentences/documents to vectorize.
        max_features (int): Number of most frequent words to use. Default is 100.

    Returns:
        numpy.ndarray: Bag-of-words embeddings.
    """
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(data)
    return X.toarray()


def data_balancer(x, y):
    label_counts = {}
    max_count = 0
    for label in y:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1
        max_count = max(max_count, label_counts[label])
    balanced_x, balanced_y = [], []
    def is_balanced():
        for count in label_counts.values():
            if count < max_count:
                return False
        return True
    label_counts = {label: 0 for label in label_counts}  # Reset counts for balancing
    while not is_balanced():
        for x_ex, y_ex in zip(x, y):
            if label_counts[y_ex] < max_count:
                balanced_x.append(x_ex)
                balanced_y.append(y_ex)
                label_counts[y_ex] += 1
    return balanced_x, np.array(balanced_y)


def get_tokenizer(text):
    """
    Tokenizes the input text into words.

    Args:
        text (str): Input text to tokenize.

    Returns:
        list: List of tokens (words).
    """
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(text)
    def preprocess(x):
        """
        Preprocesses the input text by tokenizing and padding.

        Args:
            x (str): Input text to preprocess.

        Returns:
            numpy.ndarray: Padded sequence of tokens.
        """
        sequences = tokenizer.texts_to_sequences(x)
        padded = pad_sequences(sequences, maxlen=100, padding='post')
        return padded

    return preprocess


if __name__ == "__main__":
    # Example usage
    from data_loader import load_data
    data = load_data()['text']
    embeddings = bag_of_words(data)
    print(embeddings[:10])  # Print first 10 embeddings
