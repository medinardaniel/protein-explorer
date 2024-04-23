import os
import numpy as np
from pymongo import MongoClient
import pandas as pd
from dotenv import load_dotenv

# Helper functions needed for annotations_vocab and convert_to_multi_hot
def create_annotations_vocab(labels):
    """ Create a dictionary mapping MONDO names to indices """
    label_set = set([item for sublist in labels for item in sublist])
    return {label: idx for idx, label in enumerate(label_set)}

def convert_to_multi_hot(labels, annotations_vocab):
    """ Convert labels to a multi-hot encoding format based on annotations_vocab """
    multi_hot_labels = np.zeros((len(labels), len(annotations_vocab)))
    for i, label_list in enumerate(labels):
        for label in label_list:
            if label in annotations_vocab:
                multi_hot_labels[i, annotations_vocab[label]] = 1
    return multi_hot_labels


def combine_embeddings(embedding_list, method):
    """
    Combine list of embeddings according to the specified method.
    """
    if method == 'concat':
        return np.concatenate(embedding_list, axis=0)
    elif method == 'average':
        return np.mean(embedding_list, axis=0)
    elif method == 'weighted':
        # Example weights; modify according to your specific needs
        weights = [0.7, 0.3]
        weighted_embeddings = [emb * w for emb, w in zip(embedding_list, weights)]
        return np.sum(weighted_embeddings, axis=0)
    elif method == 'min':
        return np.min(embedding_list, axis=0)
    elif method == 'max':
        return np.max(embedding_list, axis=0)
    else:
        raise ValueError("Unsupported combination method")


def fetch_data_multi(embedding_type, include_empty=True):
    """
    Fetches embeddings and optionally includes data for proteins whether they have MONDO annotations or not.

    Parameters:
    embedding_type (str): The type of embeddings to fetch (e.g., 'biobert_embeddings').
    include_empty (bool): If True, includes proteins without MONDO annotations using a placeholder.
    """
    # Load environment variables
    load_dotenv()
    MONGO_URI = os.getenv("MONGODB_URI")
    MONGO_DB = "proteinExplorer"
    MONGO_COLLECTION = "protein_embeddings"

    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]

    embeddings = []
    labels = []

    query = {embedding_type: {"$exists": True, "$ne": None}}
    if not include_empty:
        query["mondo_names"] = {"$exists": True, "$ne": []}

    projection = {embedding_type: 1, 'mondo_names': 1}

    cursor = collection.find(query, projection)
    for doc in cursor:
        if embedding_type not in doc:
            continue
        mondo_names = doc.get('mondo_names', ['No_Annotation'] if include_empty else [])
        fetched_embedding = doc.get(embedding_type)

        if fetched_embedding is not None:
            embeddings.append(fetched_embedding)
            
        labels.append(mondo_names)

    # Create vocabulary from all possible labels and convert to multi-hot
    annotations_vocab = create_annotations_vocab(labels)
    labels = convert_to_multi_hot(labels, annotations_vocab)
    embeddings = np.array(embeddings, dtype=np.float32)

    return embeddings, labels, annotations_vocab


def fetch_data(observation, target):
    """
    Fetches the values for the keys that match the observation and target strings.

    Parameters:
    observation (str): The key for the observation values to fetch.
    target (str): The key for the target values to fetch.
    """
    # Load environment variables
    load_dotenv()
    MONGO_URI = os.getenv("MONGODB_URI")
    MONGO_DB = "proteinExplorer"
    MONGO_COLLECTION = "protein_embeddings"

    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]

    # Fetch the values for the keys that match the observation and target strings
    cursor = collection.find({}, {observation: 1, target: 1, "_id": 0})

    # Create a DataFrame with the observation and target as columns
    df = pd.DataFrame(list(cursor))

    return df


def fetch_data_binary(embedding_type, mondo_name, include_empty=False):
    """
    Fetches embeddings and binary labels indicating presence or absence of the specified MONDO name.
    
    Parameters:
    embedding_type (str): The type of embedding to fetch (e.g., 'biobert_embeddings').
    mondo_name (str): The specific MONDO name to check for in the protein annotations.
    include_empty (bool): If True, include records where MONDO annotations may be empty.
    """
    # Load environment variables
    load_dotenv()
    MONGO_URI = os.getenv("MONGODB_URI")
    MONGO_DB = "proteinExplorer"
    MONGO_COLLECTION = "protein_embeddings"

    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]

    embeddings = []
    labels = []

    # Modify the query to adjust based on the include_empty parameter
    if include_empty:
        cursor = collection.find({}, {embedding_type: 1, 'mondo_names': 1})
    else:
        cursor = collection.find({f"{embedding_type}": {"$exists": True, "$ne": None}, "mondo_names": {"$exists": True, "$ne": []}}, {embedding_type: 1, 'mondo_names': 1})

    for doc in cursor:
        embedding = doc.get(embedding_type)
        mondo_names = doc.get('mondo_names', [])

        if embedding:
            embeddings.append(embedding)
            # Create a binary label indicating whether the specified MONDO name is in the mondo_names list
            labels.append(1 if mondo_name in mondo_names else 0)

    # Convert embeddings list to a NumPy array for use in models
    embeddings = np.array(embeddings, dtype=np.float32)
    
    return embeddings, labels

def get_test_data(embeddings_test):
    """
    For each document in the MongoDB database where the embedding in embeddings_test corresponds to func_embedding, save the 'function' and 'mondo_names' to a pandas DataFrame.
    :param embeddings_test: The embeddings used for testing
    :return: A pandas DataFrame containing the 'function' and 'mondo_names' for each document
    """
    # Load environment variables
    load_dotenv()
    MONGO_URI = os.getenv("MONGODB_URI")
    MONGO_DB = "proteinExplorer"
    MONGO_COLLECTION = "protein_embeddings"

    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]

    function = []
    labels = []

    for embedding in embeddings_test:
        document = collection.find_one({"func_embedding": embedding.tolist()})
        function.append(document["function"])
        labels.append(document["mondo_names"])

    data = {"function": function, "mondo_names": labels}

    return pd.DataFrame(data)


