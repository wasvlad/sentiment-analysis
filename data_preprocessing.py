from sentence_transformers import SentenceTransformer

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

if __name__ == "__main__":
    # Example usage
    from data_loader import load_data
    data = load_data()['text']
    embeddings = s_bert(data)
    print(embeddings[:10])  # Print first 10 embeddings
