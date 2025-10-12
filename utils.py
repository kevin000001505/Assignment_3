import logging
import os
from typing import List, Tuple
import numpy as np
import torch
from gensim.models import KeyedVectors

logging.basicConfig(
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("main.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def configuration():
    """Detect available compute device and number of CPU cores.

    Returns:
        tuple: (torch.device, int) device to use and CPU core count.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")
    cpu_count = os.cpu_count() or 4
    logger.info(f"Number of CPU cores available: {cpu_count}")
    return device, cpu_count


def load_embeddings_from_bin_gz(file_path: str) -> KeyedVectors:
    """Load Word2Vec embeddings from a binary (possibly compressed) file.

    Args:
        file_path (str): Path to the word2vec-format binary file.

    Returns:
        KeyedVectors: Gensim KeyedVectors object with loaded embeddings.
    """
    logger.info("Loading Word2Vec embeddings...")
    embeddings = KeyedVectors.load_word2vec_format(file_path, binary=True)
    logger.info(
        f"Loaded {len(embeddings)} word vectors with {embeddings.vector_size} dimensions"
    )
    return embeddings


def transform_data(
    data: List[Tuple[List[str], List[str]]],
    word_to_index,
    target_to_idx,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert dataset of token/tag pairs into index tensors.

    Args:
        data: List of (sentence_tokens, sentence_tags).
        word_to_index: Mapping from token -> index (must contain "<PAD>").
        target_to_idx: Mapping from tag -> index (must contain "<pad>").

    Returns:
        Tuple[torch.LongTensor, torch.LongTensor]: X tensor (batch, seq_len) and Y tensor (batch, seq_len).
    """
    X = []
    Y = []
    for sentence, targets in data:
        x_indices = [
            word_to_index.get(word, word_to_index["<PAD>"]) for word in sentence
        ]
        y_indices = [target_to_idx.get(tag, target_to_idx["<pad>"]) for tag in targets]
        X.append(x_indices)
        Y.append(y_indices)
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)


def get_embedding_matrix(word_to_index, embeddings, embedding_dim) -> torch.Tensor:
    """Build an embedding matrix aligned with word_to_index.

    For words present in the provided embeddings, their vectors are used;
    missing words receive a random normal vector.

    Args:
        word_to_index: Mapping from token -> index.
        embeddings: Gensim KeyedVectors or dict-like mapping of word -> vector.
        embedding_dim: Dimensionality of embeddings.

    Returns:
        torch.FloatTensor: Matrix of shape (vocab_size, embedding_dim).
    """
    vocab_size = len(word_to_index)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, idx in word_to_index.items():
        if word in embeddings:
            # We change the row of value to embedding by the word_to_index
            embedding_matrix[idx] = embeddings[word]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return torch.tensor(embedding_matrix, dtype=torch.float32)
