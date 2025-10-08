import os
import re
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from typing import List, Tuple
from pre_process import Preprocessor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("main.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

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


def load_embeddings_from_bin_gz(file_path: str) -> KeyedVectors:
    """Load Word2Vec embeddings from a compressed binary file."""
    logger.info("Loading Word2Vec embeddings...")
    embeddings = KeyedVectors.load_word2vec_format(file_path, binary=True)
    logger.info(
        f"Loaded {len(embeddings)} word vectors with {embeddings.vector_size} dimensions"
    )
    return embeddings


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.RNN(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.params = dict(self.rnn.named_parameters())

    def forward(self, x):
        pass


def main():

    processor = Preprocessor()
    train = processor.load_data("conll2003/train.txt")
    valid = processor.load_data("conll2003/valid.txt")
    test = processor.load_data("conll2003/test.txt")
    breakpoint()

    # Load Google's pre-trained Word2Vec embeddings
    embeddings = load_embeddings_from_bin_gz("GoogleNews-vectors-negative300.bin.gz")


if __name__ == "__main__":
    main()
