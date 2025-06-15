from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

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

def bag_of_words(data, max_features=10):
    """
    Function to generate bag-of-words embeddings from text data using only the most frequent words.

    Args:
        data (list or array-like): List of sentences/documents to vectorize.
        max_features (int): Number of most frequent words to use. Default is 1000.

    Returns:
        numpy.ndarray: Bag-of-words embeddings.
    """
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(data)
    return X.toarray()



if __name__ == "__main__":
    # Example usage
    from data_loader import load_data
    data = load_data()['text']
    embeddings = bag_of_words(data)
    print(embeddings[:10])  # Print first 10 embeddings
